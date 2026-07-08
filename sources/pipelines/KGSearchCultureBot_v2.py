"""
title: KGSearchCultureBot V2 — Neo4j Knowledge Graph + Hybrid RAG Pipeline
author: Ottobot
version: 2.0.0
description: |
    OpenWebUI Pipeline combining a Neo4j Knowledge Graph with hybrid FAISS+BM25 RAG.

    v2 changes vs v1 (KGSearchCultureBot.py):
    - Cypher generation now delegates to sc_kg_nl2cypher.py (the same module used in the
      searchculture_kg_nl2cypher.ipynb notebook), instead of this file's own hand-rolled
      schema prompt. That module's schema/few-shots were verified directly against the
      live Neo4j (db.schema.nodeTypeProperties()) and fix several real bugs the v1 prompt
      had: Place/TimeSpan/Concept only have a `label` property (no pref_label_el/en/
      alt_labels), materials need (item)-[:MADE_OF]->(Material) not item.medium_values
      (which doesn't exist), periods need -[:HAS_TEMPORAL_REFERENCE|FROM_PERIOD]->, and —
      most importantly — relationship filters must use EXISTS {{ ... }} instead of
      `OPTIONAL MATCH ... WHERE ...`, because the latter never excludes a row in Cypher
      (it silently returns the row with the optional variable set to NULL), so v1 queries
      effectively ignored their own place/period/type filters.
    - Place taxonomy is loaded once at startup (self.places) and passed into the prompt
      builder, instead of re-reading the JSON file on every request.
    - Cypher syntax validation now reuses sc_kg_nl2cypher.validate_cypher_text (regex
      word-boundary checks) instead of this file's own substring checks, which could
      false-positive (e.g. rejecting any result containing the word "subset") or
      false-negative depending on keyword placement.
    - Answer generation prompt: detailed/thorough answers when there are many good
      matches (not artificially trimmed to 2-3), clearer structure rules, and a short
      identifying signature line so the user always knows the answer came from CultureBot.
    - RAG retrieval (TOP_K_SEMANTIC/TOP_K_BM25) raised from 10 to 100 each, with explicit
      stable-hash deduplication between the FAISS and BM25 result sets before RRF fusion,
      and MAX_CONTEXT_RECORDS/MAX_CONTEXT_CHARS raised so more of those matches can
      actually reach the final answer instead of being retrieved and then discarded.

    Data sources (two separate files, two separate roles):
    - puretext_chunks.jsonl  → KG path: {id, text} records for O(1) artifact lookup by ID
    - puretext_chunks.md     → RAG path: Markdown text chunks for FAISS+BM25 semantic search

    Three query modes (QUERY_MODE valve):
    - hybrid: Both paths run; merged context → single streamed answer (default)
    - kg:     LLM generates Cypher → Neo4j traversal → JSONL artifact lookup → streamed answer
    - rag:    FAISS + BM25 search (RRF fusion) over Markdown chunks → streamed answer

    Four embedding backends (EMBEDDING_MODEL valve):
    - sentence-transformers/all-MiniLM-L6-v2  (HuggingFace, 384d, CPU-friendly, default)
    - Qwen/Qwen3-Embedding-0.6B               (HuggingFace, 1024d, requires GPU)
    - text-embedding-3-small                  (OpenAI API, 1536d)
    - text-embedding-3-large                  (OpenAI API, 3072d, best quality)

    Disk cache (FAISS index + BM25 corpus) avoids full rebuild on restart.
    Cache is keyed on embedding model + MD file mtime, so it invalidates automatically
    when the source Markdown is updated.
"""

import os
import sys
import json
import pickle
import hashlib
import logging
import time
import re
import gc
import unicodedata
from pathlib import Path
from typing import List, Generator, Optional, Dict, Any, Literal
from enum import Enum
from collections import Counter

from openai import OpenAI
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# sc_kg_nl2cypher.py is deployed into the non-scanned kg_jsons/ subdir (NOT the top-level
# pipelines dir): the OpenWebUI pipelines server imports every top-level .py as a Pipeline,
# and a helper with no Pipeline class gets quarantined behind a shadowing same-named folder.
# Add that subdir to sys.path so `from sc_kg_nl2cypher import ...` resolves to the real module.
_ASSETS_DIR = str(Path(__file__).resolve().parent / "kg_jsons")
if _ASSETS_DIR not in sys.path:
    sys.path.insert(0, _ASSETS_DIR)

from sc_kg_nl2cypher import (
    load_places,
    build_cypher_prompt,
    validate_cypher_text,
    strip_code_fences,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Embedding model registry ──────────────────────────────────────────────────
# Tradeoff summary:
#   MiniLM      — fast, free, CPU-only, good baseline quality
#   Qwen3-0.6B  — better multilingual quality, needs ~4 GB GPU VRAM
#   OpenAI small — excellent quality, costs API credits, no GPU needed
#   OpenAI large — best quality, higher API cost per embedding

class EmbeddingModel(str, Enum):
    MINILM       = "sentence-transformers/all-MiniLM-L6-v2"
    QWEN3_06B    = "Qwen/Qwen3-Embedding-0.6B"
    OPENAI_SMALL = "text-embedding-3-small"
    OPENAI_LARGE = "text-embedding-3-large"


EMBEDDING_MODEL_CONFIGS = {
    EmbeddingModel.MINILM: {
        "name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "batch_size": 64,
        "faiss_batch_size": 1000,
        "min_gpu_memory_gb": 0.5,
        "provider": "huggingface",
    },
    EmbeddingModel.QWEN3_06B: {
        "name": "Qwen3-Embedding-0.6B",
        "dimension": 1024,
        "batch_size": 2,        # small batch: 0.6B model is memory-heavy per batch on GPU
        "faiss_batch_size": 50,
        "min_gpu_memory_gb": 4.0,
        "provider": "huggingface",
    },
    EmbeddingModel.OPENAI_SMALL: {
        "name": "text-embedding-3-small",
        "dimension": 1536,
        "batch_size": 200,      # 300k token limit: 200 × ~1500 char MD chunks ≈ 75k tokens
        "faiss_batch_size": 500,
        "min_gpu_memory_gb": 0.0,
        "provider": "openai",
    },
    EmbeddingModel.OPENAI_LARGE: {
        "name": "text-embedding-3-large",
        "dimension": 3072,
        "batch_size": 200,
        "faiss_batch_size": 500,
        "min_gpu_memory_gb": 0.0,
        "provider": "openai",
    },
}


class OpenAIEmbeddingWrapper:
    """
    LangChain-compatible wrapper around the OpenAI Embeddings API.
    Supports batched embed_documents() for efficient FAISS index construction.
    """

    def __init__(self, client: OpenAI, model: str, batch_size: int = 512):
        self._client = client
        self._model = model
        self._batch_size = batch_size

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        # OpenAI rejects empty strings — replace with a single space
        texts = [t if t.strip() else " " for t in texts]
        response = self._client.embeddings.create(model=self._model, input=texts)
        # Sort by index: API does not guarantee response order
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for i in range(0, len(texts), self._batch_size):
            results.extend(self._embed_batch(texts[i: i + self._batch_size]))
        return results

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

    # Some LangChain FAISS versions call embedding_function(text) directly
    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


# ── Pipeline: cute ──────────────────────────────────────────────────────────────────

class Pipeline:
    """
    KGSearchCultureBot — Neo4j KG + Hybrid RAG OpenWebUI Pipeline.

    On first call, _initialize() loads the JSONL file into:
      records_by_id  — dict {artifact_id → {id, text}} for O(1) KG path lookup
      documents      — list of Documents for FAISS/BM25 RAG indexes (1 doc = 1 record)

    Subsequent calls hit the disk cache (FAISS index + BM25 corpus pickle) unless
    the source file or any relevant valve has changed.
    """

    name = "KGSearchCultureBot"

    class Valves(BaseModel):
        # ── OpenAI ──────────────────────────────────────────────────────────
        OPENAI_API_KEY: str = Field(
            default=os.getenv("OPENAI_API_KEY", ""),
            description="OpenAI API key. Leave blank here and set the OPENAI_API_KEY env var (see .env).",
        )
        OPENAI_MODEL: str = Field(
            default="gpt-5.5",
            description="Generation model. Options: gpt-4o | gpt-5.5 | gpt-4o-mini",
        )
        OPENAI_BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="Override for Azure endpoints or local proxies.",
        )

        # ── Neo4j ────────────────────────────────────────────────────────────
        NEO4J_URI: str = Field(
            default=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
            description="Bolt URI for the Neo4j KG instance. Set via the NEO4J_URI env var (see .env).",
        )
        NEO4J_USER: str = Field(default=os.getenv("NEO4J_USER", "neo4j"))
        NEO4J_PASSWORD: str = Field(
            default=os.getenv("NEO4J_PASSWORD", ""),
            description="Neo4j password. Leave blank here and set the NEO4J_PASSWORD env var (see .env).",
        )

        # ── Embeddings ───────────────────────────────────────────────────────
        EMBEDDING_MODEL: str = Field(
            default=EmbeddingModel.OPENAI_SMALL.value,
            description=(
                "sentence-transformers/all-MiniLM-L6-v2 (fast, CPU) | "
                "Qwen/Qwen3-Embedding-0.6B (quality, needs GPU) | "
                "text-embedding-3-small (OpenAI) | "
                "text-embedding-3-large (OpenAI, best quality)"
            ),
        )

        # ── Sources ──────────────────────────────────────────────────────────
        JSONL_PATH: str = Field(
            default="/app/pipelines/puretext_chunks.jsonl",
            description='JSONL file for the KG path. Each line: {"id": "...", "text": "..."}',
        )
        MD_PATH: str = Field(
            default="/app/pipelines/puretext_chunks.md",
            description="Markdown file for the RAG path. Chunks are split on ## headers.",
        )

        # ── Cache ────────────────────────────────────────────────────────────
        CACHE_DIR: str = Field(
            default="/app/pipelines/cache/kgsearchculturebot_v2",
            description="Directory for FAISS index and BM25 corpus cache files.",
        )
        ENABLE_CACHE: bool = Field(default=True)
        FAST_CONTENT_HASH: bool = Field(
            default=False,
            description="False = hash file content for cache key (survives docker cp). True = use mtime+size (faster but invalidated by re-copy).",
        )

        # ── Query mode ───────────────────────────────────────────────────────
        QUERY_MODE: Literal["hybrid", "kg", "rag"] = Field(
            default="hybrid",
            description=(
                "hybrid: KG + RAG merged into one prompt (single LLM call) | "
                "kg: Knowledge Graph only | rag: Vector/keyword search only"
            ),
        )

        # ── KG retrieval ─────────────────────────────────────────────────────
        MAX_GRAPH_ROWS: int = Field(
            default=20,
            description="Max rows returned per Cypher query (injected as LIMIT clause).",
        )
        MAX_CONTEXT_RECORDS: int = Field(
            default=50,
            description="Max records included in LLM context regardless of how many were retrieved. "
                        "Raised from v1's 10 so a thorough answer can actually draw on many matches; "
                        "MAX_CONTEXT_CHARS below is the real safety net against an oversized prompt.",
        )

        # ── RAG retrieval ─────────────────────────────────────────────────────
        TOP_K_SEMANTIC: int = Field(default=100, description="Top-K results from FAISS semantic search.")
        TOP_K_BM25: int = Field(default=100, description="Top-K results from BM25 keyword search.")
        SEMANTIC_WEIGHT: float = Field(
            default=0.6,
            description="RRF weight for semantic results (0–1). Remainder goes to BM25.",
        )
        SEMANTIC_SCORE_MODE: str = Field(
            default="auto",
            description="Unused since RRF replaced min-max fusion. Kept for backwards compat with valves.json.",
        )
        RRF_K: int = Field(
            default=60,
            description="Reciprocal Rank Fusion constant. Higher = ranks matter less; standard value is 60.",
        )

        # ── Context / generation ──────────────────────────────────────────────
        MAX_CONTEXT_CHARS: int = Field(
            default=45000,
            description="Hard character cap on assembled context to stay within the LLM's token window.",
        )
        TEMPERATURE: float = Field(
            default=0.2,
            description="Generation temperature. Lower = more factual/grounded.",
        )

        # ── Debug ─────────────────────────────────────────────────────────────
        DEBUG: bool = Field(default=False)

    def __init__(self):
        self.name = "KGSearchCultureBot V2 (Clean)"
        self.valves = self.Valves()

        # Lazy clients — created on first use, re-created when credentials change
        self._openai_client: Optional[OpenAI] = None
        self._neo4j_driver = None

        # KG path: O(1) lookup by artifact ID
        self.records_by_id: Dict[str, Dict[str, Any]] = {}

        # Place taxonomy for Cypher-prompt place candidates (sc_kg_nl2cypher.py) —
        # loaded once in _initialize() and reused across requests.
        self.places: List[Dict[str, Any]] = []

        # RAG path: stable ordered list for FAISS/BM25 (1 entry = 1 JSONL record)
        self.documents: List[Document] = []
        self.embeddings = None
        self.vectorstore: Optional[FAISS] = None
        self.bm25: Optional[BM25Okapi] = None

        self.initialized = False
        self.current_embedding_model: Optional[str] = None

    # ── Client management ─────────────────────────────────────────────────────

    def _get_openai_client(self) -> OpenAI:
        """Lazily create / refresh when the API key changes."""
        if (
            self._openai_client is None
            or getattr(self._openai_client, "_cached_api_key", None) != self.valves.OPENAI_API_KEY
        ):
            self._openai_client = OpenAI(
                api_key=self.valves.OPENAI_API_KEY,
                base_url=self.valves.OPENAI_BASE_URL or "https://api.openai.com/v1",
            )
            self._openai_client._cached_api_key = self.valves.OPENAI_API_KEY
        return self._openai_client

    def _chat_completion(self, **kwargs):
        """
        Wrapper around chat.completions.create that tolerates models which only accept the
        default temperature (gpt-5.x reasoning models return a 400 'temperature does not
        support <x>' for any non-default value). On that specific error, drop temperature
        and retry — answer content is unchanged, the call just stops 400ing.
        """
        client = self._get_openai_client()
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            msg = str(e).lower()
            retry = dict(kwargs)
            changed = False
            if "temperature" in msg and "temperature" in retry:
                retry.pop("temperature", None)
                changed = True
            if "verbosity" in msg and "verbosity" in retry:
                retry.pop("verbosity", None)
                changed = True
            if changed:
                return client.chat.completions.create(**retry)
            raise

    def _get_neo4j_driver(self):
        """Lazily create / reconnect when the URI changes."""
        cached_uri = getattr(self._neo4j_driver, "_cached_uri", None)
        if self._neo4j_driver is None or cached_uri != self.valves.NEO4J_URI:
            if self._neo4j_driver is not None:
                try:
                    self._neo4j_driver.close()
                except Exception:
                    pass
            self._neo4j_driver = GraphDatabase.driver(
                self.valves.NEO4J_URI,
                auth=(self.valves.NEO4J_USER, self.valves.NEO4J_PASSWORD),
            )
            self._neo4j_driver._cached_uri = self.valves.NEO4J_URI
        return self._neo4j_driver

    def shutdown(self):
        """Called by OpenWebUI when the pipeline is unloaded — close the Neo4j driver."""
        if self._neo4j_driver is not None:
            try:
                self._neo4j_driver.close()
            except Exception:
                pass

    # ── Config helpers ────────────────────────────────────────────────────────

    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        for model_enum, config in EMBEDDING_MODEL_CONFIGS.items():
            if model_enum.value == model_name:
                return config
        return {"batch_size": 32, "faiss_batch_size": 500, "min_gpu_memory_gb": 1.0, "provider": "huggingface"}

    def _get_cache_path(self) -> Path:
        model_short = re.sub(r"[^a-z0-9]", "_", self.valves.EMBEDDING_MODEL.split("/")[-1].lower())
        cache_dir = Path(self.valves.CACHE_DIR) / model_short
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _get_content_hash(self) -> str:
        """
        Cache key for the RAG index = embedding model + MD path + retrieval settings + MD file metadata.
        Keyed on MD_PATH because the FAISS/BM25 indexes are built from Markdown, not JSONL.
        """
        source_path = self.valves.MD_PATH
        hash_input = (
            f"{self.valves.EMBEDDING_MODEL}|{source_path}|"
            f"{self.valves.TOP_K_SEMANTIC}|{self.valves.TOP_K_BM25}|"
            f"{self.valves.SEMANTIC_WEIGHT}|{self.valves.SEMANTIC_SCORE_MODE}"
        )
        if os.path.exists(source_path):
            if self.valves.FAST_CONTENT_HASH:
                stat = os.stat(source_path)
                mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
                hash_input += f"|{mtime_ns}|{stat.st_size}"
            else:
                with open(source_path, "rb") as f:
                    hash_input += "|" + hashlib.md5(f.read()).hexdigest()
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    # ── Tokenization ──────────────────────────────────────────────────────────

    def _bm25_tokenize(self, text: str) -> List[str]:
        text = unicodedata.normalize("NFKC", text).lower()
        text = re.sub(r"https?://\S+", " ", text)
        # Explicit Greek diacritical chars: corpus contains Greek-language descriptions
        text = re.sub(r"[^\w\sάέήίόύώϊΐϋΰ]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split() if text else []

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_jsonl_records(self) -> None:
        """
        Read puretext_chunks.jsonl into records_by_id for O(1) KG path lookup.
        Only the KG path uses this dict — the RAG path indexes Markdown instead.

        Supports {"id": "...", "text": "..."} and {"uri": "...", "text": "..."}.
        """
        path = self.valves.JSONL_PATH
        if not os.path.exists(path):
            logger.warning(f"JSONL file not found: {path}")
            return

        self.records_by_id = {}

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    if self.valves.DEBUG:
                        logger.warning(f"Skipping invalid JSON at line {line_num}")
                    continue

                if not isinstance(obj, dict):
                    continue

                rec_id = str(obj.get("id") or obj.get("uri") or "").strip()
                rec_text = str(
                    obj.get("text") or obj.get("content") or
                    obj.get("chunk_text") or obj.get("chunk") or ""
                ).strip()

                if not rec_id or not rec_text:
                    if self.valves.DEBUG:
                        logger.warning(f"Skipping incomplete record at line {line_num}")
                    continue

                self.records_by_id[rec_id] = {"id": rec_id, "text": rec_text}

        logger.info(f"Loaded {len(self.records_by_id)} JSONL records for KG lookup")

    def _load_md_documents(self) -> None:
        """
        Read puretext_chunks.md into self.documents for FAISS/BM25 RAG indexing.
        Splits on ## headers first; falls back to recursive character splitting.
        Each chunk gets metadata: id (chunk_N), title (header text or chunk_N).
        """
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

        path = self.valves.MD_PATH
        if not os.path.exists(path):
            logger.warning(f"Markdown file not found: {path}")
            return

        self.documents = []

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
        try:
            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=False,
            )
            chunks = splitter.split_text(content)
        except Exception as e:
            logger.warning(f"Header splitting failed ({e}), falling back to character splitter")
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=1500, chunk_overlap=150,
                separators=["\n\n", "\n", " "],
            ).create_documents([content])

        for i, chunk in enumerate(chunks):
            text = chunk.page_content.strip()
            if not text:
                continue
            title = (
                chunk.metadata.get("h2") or
                chunk.metadata.get("h1") or
                chunk.metadata.get("h3") or
                f"chunk_{i}"
            )
            chunk.metadata["id"] = f"chunk_{i}"
            chunk.metadata["title"] = title
            self.documents.append(Document(page_content=text, metadata=chunk.metadata))

        logger.info(f"Loaded {len(self.documents)} Markdown chunks for RAG")

    # ── Device selection ──────────────────────────────────────────────────────

    def _get_best_device(self, min_memory_gb: float = 2.0):
        import torch
        if not torch.cuda.is_available():
            return "cpu", {"device": "cpu", "trust_remote_code": True}

        torch.cuda.empty_cache()
        best_gpu, best_free = None, 0.0
        for i in range(torch.cuda.device_count()):
            try:
                free_mem, _ = torch.cuda.mem_get_info(i)
                free_gb = free_mem / (1024 ** 3)
                logger.info(f"GPU {i}: {free_gb:.2f} GB free")
                if free_gb > best_free:
                    best_free, best_gpu = free_gb, i
            except Exception:
                continue

        if best_gpu is not None and best_free >= min_memory_gb:
            return f"cuda:{best_gpu}", {"device": f"cuda:{best_gpu}", "trust_remote_code": True}

        logger.warning("Insufficient GPU memory — falling back to CPU")
        return "cpu", {"device": "cpu", "trust_remote_code": True}

    # ── FAISS incremental build ───────────────────────────────────────────────

    def _build_faiss_incremental(self, documents: List[Document]) -> FAISS:
        """
        Build FAISS in batches to avoid OOM on large corpora.
        merge_from() is preferred over add_documents() when available because it
        uses embed_documents() (batched) rather than embed_query() per document —
        critical for OpenAI embeddings where each API call has overhead.
        """
        import torch
        model_config = self._get_model_config(self.valves.EMBEDDING_MODEL)
        batch_size = int(model_config.get("faiss_batch_size", 500))
        total_batches = (len(documents) + batch_size - 1) // batch_size
        logger.info(f"Building FAISS index: {len(documents)} docs, batch_size={batch_size}")

        vectorstore = None
        for i in range(0, len(documents), batch_size):
            batch = documents[i: i + batch_size]
            batch_num = (i // batch_size) + 1
            if batch_num % 10 == 1:
                logger.info(f"  Batch {batch_num}/{total_batches}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, self.embeddings)
            elif hasattr(vectorstore, "merge_from"):
                try:
                    tmp = FAISS.from_documents(batch, self.embeddings)
                    vectorstore.merge_from(tmp)
                except Exception:
                    vectorstore.add_documents(batch)
            else:
                vectorstore.add_documents(batch)

        return vectorstore

    # ── Cache save / load ─────────────────────────────────────────────────────

    def _save_cache(self, cache_path: Path, content_hash: str) -> None:
        try:
            if self.vectorstore:
                self.vectorstore.save_local(str(cache_path / f"faiss_{content_hash}"))
            cache_data = {
                "embedding_model": self.valves.EMBEDDING_MODEL,
                "documents": [(doc.page_content, doc.metadata) for doc in self.documents],
                "bm25_corpus": [self._bm25_tokenize(doc.page_content) for doc in self.documents],
            }
            with open(cache_path / f"cache_{content_hash}.pkl", "wb") as f:
                pickle.dump(cache_data, f)
            logger.info(f"Cache saved → {cache_path}")
        except Exception as e:
            logger.error(f"Cache save failed: {e}")

    def _load_cache(self, cache_path: Path, content_hash: str) -> bool:
        try:
            faiss_path = cache_path / f"faiss_{content_hash}"
            pickle_path = cache_path / f"cache_{content_hash}.pkl"
            if not faiss_path.exists() or not pickle_path.exists():
                return False

            self.vectorstore = FAISS.load_local(
                str(faiss_path), self.embeddings, allow_dangerous_deserialization=True
            )
            with open(pickle_path, "rb") as f:
                cache_data = pickle.load(f)

            # Reject cache built with a different embedding model
            if cache_data.get("embedding_model") != self.valves.EMBEDDING_MODEL:
                return False

            self.documents = [Document(page_content=c, metadata=m) for c, m in cache_data["documents"]]
            self.bm25 = BM25Okapi(cache_data["bm25_corpus"])
            # records_by_id is always loaded fresh from JSONL — not reconstructed from cache
            logger.info(f"Cache hit: {len(self.documents)} MD chunks")
            return True
        except Exception as e:
            logger.error(f"Cache load failed: {e}")
            return False

    # ── Initialization ────────────────────────────────────────────────────────

    def _initialize(self) -> None:
        """
        One-time setup: embedding model, JSONL loading, FAISS + BM25 indexes.
        Re-runs automatically if EMBEDDING_MODEL valve changes between requests.
        """
        if self.initialized and self.current_embedding_model == self.valves.EMBEDDING_MODEL:
            return

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Initializing KGSearchCultureBot — embedding: {self.valves.EMBEDDING_MODEL}")
        start = time.time()

        model_config = self._get_model_config(self.valves.EMBEDDING_MODEL)

        if model_config.get("provider") == "openai":
            self.embeddings = OpenAIEmbeddingWrapper(
                client=self._get_openai_client(),
                model=self.valves.EMBEDDING_MODEL,
                batch_size=int(model_config.get("batch_size", 512)),
            )
        else:
            device, model_kwargs = self._get_best_device(model_config.get("min_gpu_memory_gb", 2.0))
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.valves.EMBEDDING_MODEL,
                model_kwargs=model_kwargs,
                encode_kwargs={"normalize_embeddings": True, "batch_size": model_config["batch_size"]},
            )
        self.current_embedding_model = self.valves.EMBEDDING_MODEL

        # KG path: always load JSONL records fresh (fast, no cache needed)
        self._load_jsonl_records()

        # KG path: place taxonomy for Cypher-prompt candidates, loaded once and reused
        # (sc_kg_nl2cypher.build_cypher_prompt re-reads disk every call if this is omitted).
        self.places = load_places()
        logger.info(f"Loaded {len(self.places)} places for Cypher prompt candidates")

        # RAG path: build FAISS+BM25 from Markdown, using disk cache
        cache_path = self._get_cache_path()
        content_hash = self._get_content_hash()

        if self.valves.ENABLE_CACHE and self._load_cache(cache_path, content_hash):
            self.initialized = True
            logger.info(
                f"Ready from cache in {time.time() - start:.2f}s — "
                f"{len(self.documents)} MD chunks, {len(self.records_by_id)} KG records"
            )
            return

        self._load_md_documents()
        if not self.documents:
            logger.warning("No MD chunks loaded — check MD_PATH valve. KG path still available.")
            self.initialized = True
            return

        self.vectorstore = self._build_faiss_incremental(self.documents)
        logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi([self._bm25_tokenize(doc.page_content) for doc in self.documents])

        if self.valves.ENABLE_CACHE:
            self._save_cache(cache_path, content_hash)

        self.initialized = True
        logger.info(
            f"Ready in {time.time() - start:.2f}s — "
            f"{len(self.documents)} MD chunks, {len(self.records_by_id)} KG records"
        )

    # ── KG pipeline ───────────────────────────────────────────────────────────

    # Controlled historical-period vocabulary actually present in Neo4j's TimeSpan.label
    # field. sc_kg_nl2cypher.build_cypher_prompt() doesn't include this (it's bot-specific
    # grounding, not part of the shared notebook prompt), so it's appended on top below.
    _HISTORICAL_PERIODS_APPENDIX = """
Controlled historical periods vocabulary actually present in Neo4j (TimeSpan.label, combined Greek/English).
When the user mentions a historical period, first map it to one of the following labels. Each label may
appear as English / Greek or Greek / English. Accept spelling errors, missing Greek accents,
lowercase/uppercase differences, and common aliases. If the user's term maps to a broader period, search
that broader period directly. If ambiguous, prefer the closest controlled-vocabulary match and include
OR conditions for plausible variants.

- Aceramic Period / Προκεραμική Περίοδος
- Archaic Period / Αρχαϊκή περίοδος
- Civil War / Εμφύλιος Πόλεμος
- Early Geometric Period / Πρώιμη Γεωμετρική περίοδος
- Early Hellenistic Period / Πρώιμη Ελληνιστική περίοδος
- Establishment of New Hellenic State / Ίδρυση Νέου Ελληνικού Κράτους
- Geometric Period / Γεωμετρική περίοδος
- Hellenistic Period / Ελληνιστική περίοδος
- Interwar period / Μεσοπόλεμος
- Late Byzantine Period / Ύστερη Βυζαντινή Περίοδος
- Late Geometric Period / Ύστερη Γεωμετρική περίοδος
- Mesolithic Period / Μεσολιθική Εποχή
- Middle Archaic Period / Μέση Αρχαϊκή Περίοδος
- Middle Classical Period / Μέση Κλασική περίοδος
- Middle Geometric Period / Μέση Γεωμετρική περίοδος
- Middle Hellenistic Period / Μέση Ελληνιστική περίοδος
- Middle Neolithic Period / Μέση Νεολιθική Περίοδος
- Military junta / Δικτατορία
- Modern Greece / Νεότερη Ελλάδα
- Ottoman Period / Οθωμανική περίοδος
- Postwar Greece / Μεταπολεμική Ελλάδα
- Regime change / Μεταπολίτευση
- Reign of King George I / Βασιλεία Γεωργίου Α’
- Reign of King Otto / Βασιλεία Όθωνα
- World War I and Asia Minor Campaign / Α’ Παγκόσμιος Πόλεμος και Μικρασιατική Εκστρατεία
- World War II / Β’ Παγκόσμιος πόλεμος
- Ύστερη Αρχαϊκή Περίοδος / Late Archaic Period
- Ύστερη Ελληνιστική περίοδος / Late Hellenistic Period
- Ύστερη Εποχή του Χαλκού / Late Bronze Age
- Ύστερη Κλασική περίοδος / Late Classical Period
- Ύστερη Νεολιθική Περίοδος / Late Neolithic Period
- Βυζαντινή περίοδος / Byzantine Period
- Εποχή του Χαλκού / Bronze Age
- Κλασική περίοδος / Classical Period
- Μέση Βυζαντινή Περίοδος / Middle Byzantine Period
- Μέση Εποχή του Χαλκού / Middle Bronze Age
- Νεολιθική Περίοδος / Neolithic Period
- Πρωτογεωμετρική περίοδος / Protogeometric Period
- Πρώιμη Αρχαϊκή Περίοδος / Early Archaic Period
- Πρώιμη Βυζαντινή Περίοδος / Early Byzantine Period
- Πρώιμη Εποχή του Χαλκού / Early Bronze Age
- Πρώιμη Κλασική Περίοδος / Early Classical Period
- Πρώιμη Νεολιθική Περίοδος / Early Neolithic Period
- Ρωμαϊκή περίοδος / Roman Period

Greek tonality (τόνοι) — CRITICAL: The data in Neo4j is stored with proper Greek accents
(e.g. "Ηράκλειο", "Αθήνα", "Θεσσαλονίκη", "Όλυμπος", "Κνωσός"). If the user's question contains
Greek words written WITHOUT accents (e.g. "ηρακλειο", "αθηνα", "θεσσαλονικη", "ολυμπος", "κνωσος"),
restore the proper accented form, then search using CONTAINS conditions covering BOTH the user's
original spelling AND the accented form. This applies to place names, period names, artifact types,
and any other Greek term. If unsure, include both variants.
""".strip()

    def _generate_cypher(self, user_query: str) -> str:
        """
        Delegates schema/filters/place-hierarchy/few-shots to sc_kg_nl2cypher.build_cypher_prompt
        — the same module the searchculture_kg_nl2cypher.ipynb notebook uses, verified directly
        against the live Neo4j schema (see that module's SCHEMA_SUMMARY for what changed vs v1:
        Place/TimeSpan/Concept only have `label`, no pref_label_el/en/alt_labels; materials go
        through MADE_OF->Material, not item.medium_values which doesn't exist; periods need both
        HAS_TEMPORAL_REFERENCE and FROM_PERIOD; and relationship filters use EXISTS {{ ... }}
        instead of OPTIONAL MATCH ... WHERE ..., which never actually excludes a row in Cypher).
        The controlled historical-periods vocabulary and Greek accent-restoration guidance below
        are bot-specific extras not part of that shared prompt.
        """
        prompt = (
            build_cypher_prompt(user_query, limit=self.valves.MAX_GRAPH_ROWS, places=self.places)
            + "\n\n" + self._HISTORICAL_PERIODS_APPENDIX
            + f"\n\nReminder — generate Cypher for this exact user question:\n{user_query}"
        )
        # temperature=0.0: Cypher must be syntactically exact — no creative variation
        resp = self._chat_completion(
            model=self.valves.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise Cypher query generator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        cypher = strip_code_fences(resp.choices[0].message.content)
        ok, msg = validate_cypher_text(cypher)
        if not ok:
            raise ValueError(f"{msg}\n{cypher}")
        return msg

    def _check_cypher_with_explain(self, cypher: str) -> bool:
        """
        EXPLAIN parses and plans the query without executing it — catches invalid
        label/property names and syntax errors cheaply before touching any data.
        """
        try:
            with self._get_neo4j_driver().session() as session:
                session.run(f"EXPLAIN {cypher}")
            return True
        except Exception as e:
            if self.valves.DEBUG:
                logger.warning(f"Cypher EXPLAIN failed: {e}")
            return False

    def _is_valid_cypher(self, cypher: str) -> bool:
        ok, _ = validate_cypher_text(cypher)
        return ok and self._check_cypher_with_explain(cypher)

    def _run_cypher(self, cypher: str) -> List[Dict[str, Any]]:
        if not self._is_valid_cypher(cypher):
            raise ValueError("Generated Cypher failed validation.")

        def _read(tx):
            return [dict(r) for r in tx.run(cypher)]

        with self._get_neo4j_driver().session() as session:
            rows = session.execute_read(_read)

        if self.valves.DEBUG:
            logger.info(f"Cypher returned {len(rows)} rows")
            for row in rows[:3]:
                logger.info(f"  {row}")
        return rows

    def _extract_item_ids(self, graph_rows: List[Dict[str, Any]]) -> List[str]:
        ids: List[str] = []
        for row in graph_rows:
            item_id = row.get("item_id")
            if item_id and item_id not in ids:
                ids.append(item_id)
        return ids

    def _lookup_records_by_ids(self, item_ids: List[str]) -> List[Dict[str, Any]]:
        seen: set = set()
        matched: List[Dict[str, Any]] = []
        for item_id in item_ids:
            if item_id in seen:
                continue
            seen.add(item_id)
            record = self.records_by_id.get(item_id)
            if record is not None:
                matched.append(record)
        if self.valves.DEBUG:
            logger.info(f"KG lookup: matched {len(matched)}/{len(item_ids)} IDs to JSONL records")
        return matched

    # ── RAG pipeline ──────────────────────────────────────────────────────────

    def _doc_dedup_key(self, doc: Document) -> str:
        """
        Stable identity for a chunk across both retrievers, so the same chunk surfaced by
        both FAISS and BM25 is merged into one scored entry instead of appearing twice.
        Falls back to a content hash (not Python's randomized hash()) when no id metadata
        is present, so the key is reproducible across calls/processes.
        """
        doc_id = doc.metadata.get("id")
        if doc_id:
            return str(doc_id)
        return hashlib.md5(doc.page_content.strip().encode("utf-8")).hexdigest()

    def _rag_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Hybrid FAISS + BM25 search with Reciprocal Rank Fusion (RRF).

        RRF score per document = Σ_i  weight_i × 1 / (RRF_K + rank_i)
        where rank_i is the doc's 0-based rank in retriever i, and the weights
        are SEMANTIC_WEIGHT (FAISS) and 1 − SEMANTIC_WEIGHT (BM25).

        RRF is rank-based, so it is robust to:
          • small result sets (no min-max collapse to 0)
          • different score scales (L2 distance vs BM25 score)
          • outlier scores from one retriever

        Deduplication: both loops below key into the SAME doc_scores/doc_map by
        _doc_dedup_key(), so a chunk retrieved by both FAISS and BM25 is fused into one
        entry (its scores summed) rather than listed twice in the final context.
        """
        rrf_k = max(1, int(self.valves.RRF_K or 60))

        semantic_results = []
        if self.vectorstore:
            semantic_results = self.vectorstore.similarity_search_with_score(
                query, k=self.valves.TOP_K_SEMANTIC
            )

        bm25_results = []
        if self.bm25:
            q_tokens = self._bm25_tokenize(query)
            if q_tokens:
                scores = self.bm25.get_scores(q_tokens)
                top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.valves.TOP_K_BM25]
                bm25_results = [(self.documents[i], float(scores[i])) for i in top_idx]

        doc_scores: Counter = Counter()
        doc_map: Dict[str, Document] = {}
        sem_w = float(self.valves.SEMANTIC_WEIGHT)
        bm_w = 1.0 - sem_w

        for rank, (doc, _) in enumerate(semantic_results):
            key = self._doc_dedup_key(doc)
            doc_scores[key] += sem_w * (1.0 / (rrf_k + rank))
            doc_map[key] = doc

        for rank, (doc, _) in enumerate(bm25_results):
            key = self._doc_dedup_key(doc)
            doc_scores[key] += bm_w * (1.0 / (rrf_k + rank))
            doc_map[key] = doc

        ranked = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
        results = []
        for key in ranked:
            doc = doc_map.get(key)
            if doc is not None:
                results.append({
                    "id": key,
                    "title": doc.metadata.get("title", key),
                    "text": doc.page_content,
                    "score": float(doc_scores[key]),
                })

        if self.valves.DEBUG:
            logger.info(f"RAG: {len(results)} fused results (RRF, k={rrf_k})")
            for r in results[:3]:
                logger.info(f"  {r['id']} score={r['score']:.4f}")
        return results

    # ── Context assembly ──────────────────────────────────────────────────────

    def _build_kg_context(self, matched_records: List[Dict[str, Any]], graph_rows: List[Dict[str, Any]]) -> str:
        parts = ["## Graph results"]
        parts.append("\n".join(
            f"[Graph Row {i}] {json.dumps(row, ensure_ascii=False)}"
            for i, row in enumerate(graph_rows[: self.valves.MAX_CONTEXT_RECORDS], start=1)
        ))
        parts.append("\n## Matched records")
        total_chars = 0
        for i, rec in enumerate(matched_records[: self.valves.MAX_CONTEXT_RECORDS], start=1):
            block = f"[Record {i} | id={rec['id']}]\n{rec['text']}\n"
            if total_chars + len(block) > self.valves.MAX_CONTEXT_CHARS:
                break
            parts.append(block)
            total_chars += len(block)
        return "\n\n".join(parts)

    def _build_rag_context(self, rag_records: List[Dict[str, Any]]) -> str:
        parts = ["## RAG retrieved chunks"]
        total_chars = 0
        for i, rec in enumerate(rag_records[: self.valves.MAX_CONTEXT_RECORDS], start=1):
            label = rec.get("title") or rec.get("id", f"chunk_{i}")
            block = f"[Chunk {i} | {label} | score={rec['score']:.4f}]\n{rec['text']}\n"
            if total_chars + len(block) > self.valves.MAX_CONTEXT_CHARS:
                break
            parts.append(block)
            total_chars += len(block)
        return "\n\n".join(parts)

    def _build_hybrid_context(
        self,
        graph_rows: List[Dict[str, Any]],
        kg_records: List[Dict[str, Any]],
        rag_records: List[Dict[str, Any]],
    ) -> str:
        return (
            self._build_kg_context(kg_records, graph_rows)
            + "\n\n" + "=" * 80 + "\n\n"
            + self._build_rag_context(rag_records)
        )

    # ── Streaming answer generation ───────────────────────────────────────────

    def _generate_response(self, query: str, context: str) -> Generator[str, None, None]:
        """
        Stream the final answer via OpenAI streaming API.
        temperature=0.2: allows natural phrasing variation while keeping facts grounded.
        The system prompt enforces language matching, URI citation, and grounding rules.
        """
        has_context = bool(context.strip())

        if has_context:
            grounding_rules = (
                "2. **Grounding**: Cite artifact URIs as links — e.g. ([τίτλος/title](URI)) — for every "
                "   distinct artifact you mention. Cite as many genuinely relevant sources as the evidence "
                "   supports; do not under-cite or settle for one or two links when more relevant artifacts "
                "   are available in the context.\n"
                "3. **Write as an expert, not a search report**: Never reference the retrieval process "
                "   itself. Do NOT use phrases like 'based on the retrieved context', 'according to the "
                "   retrieved objects', 'other retrieved items were not related to...', 'from the provided "
                "   context', or any similar meta-commentary about searching/retrieving. State facts "
                "   directly, as something you know. Never describe or list items that are NOT relevant — "
                "   silently omit them instead; only ever mention artifacts that are actually relevant.\n"
                "4. **Depth and length**: Write a substantial, thorough answer — this is the default "
                "   expectation, not a fallback reserved for 'many matches'. Cover the relevant artifacts "
                "   in real detail (dates, places, materials, descriptions, subjects), draw connections "
                "   between them, and do not compress to a short list when the evidence supports more. "
                "   Only write a short answer when the evidence is genuinely thin — never pad, but always "
                "   use what's there.\n"
                "5. **Analysis**: Do not just enumerate items — analyze them. Note patterns across the "
                "   results (shared periods, places, materials, types), explain significance or historical "
                "   context, and synthesize an interpretation rather than a raw catalogue.\n"
                "6. **Honesty**: If overall coverage of the topic is thin, you may say so briefly and in "
                "   general terms — but never list or describe specific unrelated items to illustrate it. "
                "   Add general knowledge only if clearly labelled *(general knowledge)*.\n"
                "7. **Never invent**: Do not fabricate URIs, titles, dates, places, or descriptions.\n"
            )
        else:
            grounding_rules = (
                "2. No records were found in the dataset. Answer from general knowledge and state clearly "
                "   that no dataset records were retrieved.\n"
            )

        system_prompt = (
            "You are CultureBot, an expert assistant for a cultural heritage dataset "
            "backed by a Neo4j Knowledge Graph and a hybrid vector/keyword search index.\n\n"
            "You are a specialized digital assistant for searching, understanding, and presenting information related to movable monuments and cultural heritage records. Your identity is grounded in the collections of movable monuments of the National Archive of Monuments, as well as their enriched records from SearchCulture. "
            "Your role is to help users find, interpret, and make use of reliable information about cultural heritage objects by providing clear, well-documented, and understandable answers. You respond with accuracy, neutrality, and respect for cultural content, prioritizing the available official and enriched data sources.\n\n"
            "## Rules\n"
            "1. **Language**: Respond in the same language as the user's question. "
            "   Greek → Greek, English → English. Never switch mid-answer.\n"
            + grounding_rules +
            "8. **Structure**: Organize with Markdown headings (grouping by place/period/theme as relevant "
            "   to the question), a short bullet list per artifact covering its key facts, and bold for "
            "   key terms. The answer should read as a clean, well-organized piece of writing.\n"
            "9. **Summary**: Always end with a short closing section — '## Σύνοψη' for Greek answers, "
            "   '## Summary' for English — that synthesizes the key findings in a few sentences.\n"
            "10. **No follow-ups**: Do NOT end the response with follow-up question suggestions, "
            "    prompts the user could ask next, 'you might also want to know…', 'further questions', "
            "    'feel free to ask…', or any similar prompting hooks. Stop right after the summary.\n"
            "11. **Signature**: Begin every answer with a short identifying line, on its own, exactly: "
            "    '**🏛️ CultureBot**' — then a blank line, then the answer itself. This is a static "
            "    identity marker, not a follow-up prompt, and must appear even on short answers."
        )

        # Verbosity is hardcoded HIGH here in the FINAL-ANSWER path ONLY. It is deliberately
        # confined to _generate_response and is NOT referenced in _generate_cypher or in any
        # RAG retrieval code, so it cannot affect Cypher generation or what RAG retrieves —
        # it only makes this final synthesized answer longer/more detailed.
        verbosity_directive = (
            "Write at length: produce a comprehensive, in-depth answer. Develop every relevant "
            "point fully, elaborate on context and significance, and use all the supporting "
            "evidence the retrieved context offers. Favour thoroughness over brevity (while never "
            "padding with empty filler or repeating yourself)."
        )

        user_prompt = (
            f"## Retrieved context\n\n{context}\n\n---\n\n"
            f"## Question\n{query}\n\n"
            f"## Answer length\n{verbosity_directive}\n\n"
            f"*(Answer in the same language as the question above)*"
        )

        try:
            stream = self._chat_completion(
                model=self.valves.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=float(self.valves.TEMPERATURE),
                verbosity="high",
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            yield f"\n\n⚠️ OpenAI API error: {e}"

    # ── Query routing ─────────────────────────────────────────────────────────

    def _ask_kg(self, query: str) -> Generator[str, None, None]:
        try:
            cypher = self._generate_cypher(query)
        except Exception as e:
            yield f"⚠️ Cypher generation failed: {e}"
            return

        try:
            graph_rows = self._run_cypher(cypher)
        except Exception as e:
            yield f"⚠️ Cypher execution failed: {e}"
            return

        item_ids = self._extract_item_ids(graph_rows)
        matched_records = self._lookup_records_by_ids(item_ids)

        if not matched_records and not graph_rows:
            yield "No results returned by the Knowledge Graph for this query."
            return

        context = self._build_kg_context(matched_records, graph_rows)
        yield from self._generate_response(query, context)

    def _ask_rag(self, query: str) -> Generator[str, None, None]:
        rag_records = self._rag_search(query)
        if not rag_records:
            yield "No relevant records found in the RAG index."
            return
        context = self._build_rag_context(rag_records)
        yield from self._generate_response(query, context)

    def _ask_hybrid(self, query: str) -> Generator[str, None, None]:
        """
        Single-call hybrid: retrieve from both the Knowledge Graph and the RAG
        index, merge into one context, stream a single answer.

        Empirically outperforms a multi-stage pipeline because the LLM sees the
        raw retrieved evidence directly — no intermediate summarization losses,
        no forced output schema, no contradictions to reconcile between
        pre-baked sub-answers.
        """
        yield "_Searching Knowledge Graph..._\n\n"
        cypher: Optional[str] = None
        graph_rows: List[Dict[str, Any]] = []
        kg_records: List[Dict[str, Any]] = []
        try:
            cypher = self._generate_cypher(query)
            graph_rows = self._run_cypher(cypher)
            item_ids = self._extract_item_ids(graph_rows)
            kg_records = self._lookup_records_by_ids(item_ids)
        except Exception as e:
            if self.valves.DEBUG:
                yield f"⚠️ KG path failed ({e}), continuing with RAG only.\n\n"

        yield "_Searching RAG index..._\n\n"
        rag_records = self._rag_search(query)

        has_kg = bool(graph_rows or kg_records)
        has_rag = bool(rag_records)

        if not has_kg and not has_rag:
            yield "No results from either the Knowledge Graph or the RAG index."
            return

        # Merge available contexts. The single-call hybrid lets the LLM synthesise
        # without a forced three-part schema or a separate judge prompt.
        if has_kg and has_rag:
            context = self._build_hybrid_context(graph_rows, kg_records, rag_records)
        elif has_kg:
            context = self._build_kg_context(kg_records, graph_rows)
        else:
            context = self._build_rag_context(rag_records)

        yield from self._generate_response(query, context)

    # ── Pipeline entry point ──────────────────────────────────────────────────

    def pipe(
        self,
        user_message: str,
        model_id: str = None,
        messages: List[dict] = None,
        body: dict = None,
    ) -> Generator[str, None, None]:
        """
        OpenWebUI calls this method for every user message.
        On first call (or after EMBEDDING_MODEL changes) _initialize() builds the indexes.
        Dispatches to the appropriate retrieval path based on the QUERY_MODE valve.
        """
        if not (self.initialized and self.current_embedding_model == self.valves.EMBEDDING_MODEL):
            yield "⏳ Initializing index — first run may take a few minutes...\n\n"
        self._initialize()

        # Determine mode: LibreChat preset injects [mode:xxx] into the system message.
        # Fall back to the QUERY_MODE valve if no tag is found.
        mode = (self.valves.QUERY_MODE or "hybrid").strip().lower()
        if messages:
            for msg in messages:
                if msg.get("role") == "system":
                    m = re.search(
                        r'\[mode:(hybrid|kg|rag)\]',
                        msg.get("content", ""),
                        re.IGNORECASE,
                    )
                    if m:
                        mode = m.group(1).lower()
                    break

        if mode == "kg" and not self.records_by_id:
            yield "❌ No JSONL records loaded for the KG path. Check JSONL_PATH in valves and restart."
            return
        if mode == "rag" and not self.documents:
            yield "❌ No Markdown chunks loaded for the RAG path. Check MD_PATH in valves and restart."
            return
        if mode == "hybrid" and not self.records_by_id and not self.documents:
            yield "❌ Neither JSONL nor Markdown data loaded. Check JSONL_PATH and MD_PATH in valves."
            return

        if mode == "kg":
            yield from self._ask_kg(user_message)
        elif mode == "rag":
            yield from self._ask_rag(user_message)
        elif mode == "hybrid":
            yield from self._ask_hybrid(user_message)
        else:
            yield f"❌ Unknown QUERY_MODE '{mode}'. Valid options: hybrid | kg | rag"
