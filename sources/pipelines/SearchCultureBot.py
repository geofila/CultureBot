"""
title: CultureBot — OpenAI GPT + Hybrid RAG (100 results, grounded, multilingual)
author: Ottobot (patched)
version: 2.0.0
description: |
    Hybrid RAG pipeline for OpenWebUI Pipelines.
    Loads either Markdown or JSONL from SOURCE_PATH (auto-detect or forced).
    Improvements over v1:
    - Uses OpenAI ChatGPT API instead of Ollama (set OPENAI_API_KEY)
    - Retrieves top-100 results (50 semantic + 50 BM25) for maximum coverage
    - Large context window (up to 128 000 chars) to include all evidence
    - Generation prompt: detailed, synthesised answers, always grounded with URIs
    - Response language automatically matches the query language
    - Markdown chunking aligned to headings
    - BM25 tokenization (Greek-friendly)
    - Cache hash uses mtime_ns (avoids stale cache on fast edits)
"""

import os
import json
import pickle
import hashlib
import logging
import time
import re
import gc
import unicodedata
from pathlib import Path
from typing import List, Generator, Optional, Dict, Any
from enum import Enum
from collections import Counter

import requests
from openai import OpenAI
from pydantic import BaseModel, Field

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel(str, Enum):
    MINILM = "sentence-transformers/all-MiniLM-L6-v2"
    QWEN3_06B = "Qwen/Qwen3-Embedding-0.6B"
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
        "batch_size": 2,
        "faiss_batch_size": 50,
        "min_gpu_memory_gb": 4.0,
        "provider": "huggingface",
    },
    EmbeddingModel.OPENAI_SMALL: {
        "name": "text-embedding-3-small",
        "dimension": 1536,
        "batch_size": 512,   # OpenAI API supports large batches
        "faiss_batch_size": 512,
        "min_gpu_memory_gb": 0.0,
        "provider": "openai",
    },
    EmbeddingModel.OPENAI_LARGE: {
        "name": "text-embedding-3-large",
        "dimension": 3072,
        "batch_size": 512,
        "faiss_batch_size": 512,
        "min_gpu_memory_gb": 0.0,
        "provider": "openai",
    },
}


class OpenAIEmbeddingWrapper:
    """
    Langchain-compatible embeddings wrapper that calls the OpenAI Embeddings API.
    Supports batch embedding for FAISS index building.
    """

    def __init__(self, client: OpenAI, model: str, batch_size: int = 512):
        self._client = client
        self._model = model
        self._batch_size = batch_size

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        # OpenAI requires non-empty strings
        texts = [t if t.strip() else " " for t in texts]
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for i in range(0, len(texts), self._batch_size):
            results.extend(self._embed_batch(texts[i : i + self._batch_size]))
        return results

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

    # Make the wrapper compatible with LangChain FAISS versions that expect
    # a callable `embedding_function(text) -> vector`.
    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


class Pipeline:
    """SearchCultureBot RAG Pipeline — OpenAI GPT backend, 100-result hybrid retrieval."""

    class Valves(BaseModel):
        # ── OpenAI ──────────────────────────────────────────────────────────
        OPENAI_API_KEY: str = Field(
            default=os.getenv("OPENAI_API_KEY", ""),
            description="OpenAI API key. Leave blank here and set the OPENAI_API_KEY env var (see .env).",
        )
        OPENAI_MODEL: str = Field(default=os.getenv("OPENAI_MODEL", "gpt-5.4"))
        OPENAI_BASE_URL: str = Field(default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))  # override for Azure / proxy

        # ── Embeddings ──────────────────────────────────────────────────────
        EMBEDDING_MODEL: str = Field(default=EmbeddingModel.OPENAI_SMALL.value)

        # ── Source selection ─────────────────────────────────────────────────
        JSON_PATH: str = Field(default="/app/pipelines/puretext_chunks.jsonl")
        SOURCE_PATH: str = Field(default="/app/pipelines/puretext_chunks.md")
        SOURCE_FORMAT: str = Field(default="auto")  # auto|jsonl|markdown

        # ── Cache ────────────────────────────────────────────────────────────
        CACHE_DIR: str = Field(default="/app/pipelines/cache/puretext_chunks")
        ENABLE_CACHE: bool = Field(default=True)
        FAST_CONTENT_HASH: bool = Field(default=True)

        # ── Parsing / chunking ───────────────────────────────────────────────
        PARSE_JSON_CHUNK_TEXT: bool = Field(default=True)
        MAX_CHARS_PER_RECORD: int = Field(default=1200)
        TARGET_CHUNKS: int = Field(default=400)

        # ── Retrieval — 50+50 = 100 results ──────────────────────────────────
        TOP_K_SEMANTIC: int = Field(default=50)
        TOP_K_BM25: int = Field(default=50)
        SEMANTIC_WEIGHT: float = Field(default=0.6)
        SEMANTIC_SCORE_MODE: str = Field(
            default="auto",
            description="auto|distance|similarity. FAISS usually returns distance (lower is better)."
        )

        # ── Prompt / context ─────────────────────────────────────────────────
        MAX_CONTEXT_LENGTH: int = Field(default=128000)  # GPT-4o supports 128k tokens
        TEMPERATURE: float = Field(default=0.2)          # lower = more factual / grounded

        # ── Grounding ────────────────────────────────────────────────────────
        STRICT_GROUNDED_ANSWER: bool = Field(default=True)
        NO_EVIDENCE_MESSAGE: str = Field(
            default="I could not find enough evidence in the indexed dataset to answer this reliably."
        )

        # ── Debugging ────────────────────────────────────────────────────────
        DEBUG: bool = Field(default=False)

    def __init__(self):
        self.name = "CultureBot"
        self.valves = self.Valves()
        self.documents: List[Document] = []
        self.vectorstore: Optional[FAISS] = None
        self.bm25: Optional[BM25Okapi] = None
        self.embeddings = None  # HuggingFaceEmbeddings or OpenAIEmbeddingWrapper
        self.initialized = False
        self.current_embedding_model: Optional[str] = None
        self._openai_client: Optional[OpenAI] = None

    def _get_openai_client(self) -> OpenAI:
        """Lazily create / refresh the OpenAI client when the key or base URL changes."""
        if (
            self._openai_client is None
            or getattr(self._openai_client, "_api_key", None) != self.valves.OPENAI_API_KEY
        ):
            self._openai_client = OpenAI(
                api_key=self.valves.OPENAI_API_KEY,
                base_url=self.valves.OPENAI_BASE_URL or "https://api.openai.com/v1",
            )
            self._openai_client._api_key = self.valves.OPENAI_API_KEY  # cache for comparison
        return self._openai_client

    # ---------------------------
    # Helpers
    # ---------------------------
    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        for model_enum, config in EMBEDDING_MODEL_CONFIGS.items():
            if model_enum.value == model_name:
                return config
        return {"batch_size": 32, "faiss_batch_size": 500, "min_gpu_memory_gb": 1.0}

    def _get_cache_path(self) -> Path:
        model_short = self.valves.EMBEDDING_MODEL.split("/")[-1].lower()
        model_short = re.sub(r"[^a-z0-9]", "_", model_short)
        cache_dir = Path(self.valves.CACHE_DIR) / model_short
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _get_source_path(self) -> str:
        # Prefer SOURCE_PATH; fallback to JSON_PATH
        sp = (self.valves.SOURCE_PATH or "").strip()
        return sp if sp else self.valves.JSON_PATH

    def _detect_source_format(self, source_path: str) -> str:
        fmt = (self.valves.SOURCE_FORMAT or "auto").strip().lower()
        if fmt in {"jsonl", "markdown"}:
            return fmt
        if source_path.lower().endswith((".md", ".markdown")):
            return "markdown"
        return "jsonl"

    def _get_content_hash(self) -> str:
        """
        Cache key includes:
        - embedding model
        - source path + format
        - relevant chunking/parsing parameters
        - file metadata or content hash
        """
        source_path = self._get_source_path()
        source_format = self._detect_source_format(source_path)

        hash_input = (
            f"{self.valves.EMBEDDING_MODEL}|{source_path}|{source_format}|"
            f"{self.valves.PARSE_JSON_CHUNK_TEXT}|{self.valves.MAX_CHARS_PER_RECORD}|"
            f"{self.valves.TARGET_CHUNKS}|{self.valves.TOP_K_SEMANTIC}|{self.valves.TOP_K_BM25}|"
            f"{self.valves.SEMANTIC_WEIGHT}|{self.valves.SEMANTIC_SCORE_MODE}"
        )

        if os.path.exists(source_path):
            if self.valves.FAST_CONTENT_HASH:
                stat = os.stat(source_path)
                # Use nanosecond mtime to avoid stale cache on rapid edits
                mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
                hash_input += f"|{mtime_ns}|{stat.st_size}"
            else:
                with open(source_path, "rb") as f:
                    hash_input += "|" + hashlib.md5(f.read()).hexdigest()

        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    # ---------------------------
    # Tokenization (BM25)
    # ---------------------------
    def _bm25_tokenize(self, text: str) -> List[str]:
        """
        Better BM25 tokenization:
        - normalize unicode (Greek/diacritics stable)
        - drop URLs
        - remove punctuation (keep word chars + Greek letters)
        """
        text = unicodedata.normalize("NFKC", text).lower()
        text = re.sub(r"https?://\S+", " ", text)  # drop URLs
        # Keep letters/digits/underscore/whitespace + common Greek accented letters
        text = re.sub(r"[^\w\sάέήίόύώϊΐϋΰ]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        return text.split()

    # ---------------------------
    # JSONL parsing
    # ---------------------------
    def _extract_jsonl_record_text(self, item: Any) -> str:
        if isinstance(item, str):
            return item.strip()
        if not isinstance(item, dict):
            return str(item).strip()

        uri = str(item.get("uri", "")).strip()
        raw = item.get("chunk_text") or item.get("content") or item.get("text") or item.get("chunk")
        if raw is None:
            return str(item)
        if not isinstance(raw, str):
            return str(raw)

        text_parts: List[str] = []
        title = ""
        desc = ""
        created = ""
        place = ""
        subjects: List[str] = []
        labels: List[str] = []

        if self.valves.PARSE_JSON_CHUNK_TEXT:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    for node in parsed:
                        if not isinstance(node, dict):
                            continue
                        node_type = str(node.get("@type", "")).lower()

                        if "providedcho" in node_type:
                            t = node.get("title")
                            if isinstance(t, list):
                                vals = [x.get("@value") for x in t if isinstance(x, dict) and x.get("@value")]
                                if vals:
                                    title = vals[0]

                            d = node.get("description")
                            if isinstance(d, dict):
                                desc = str(d.get("@value", "")).strip()
                            elif isinstance(d, str):
                                desc = d.strip()

                            created = str(node.get("created", "")).strip()

                            spatial = node.get("dcterms:spatial") or node.get("spatial")
                            if isinstance(spatial, dict):
                                place = str(spatial.get("@value", "")).strip()
                            elif isinstance(spatial, str):
                                place = spatial.strip()

                            subj = node.get("subject")
                            if isinstance(subj, list):
                                subjects.extend([str(s) for s in subj[:12]])
                            elif isinstance(subj, str):
                                subjects.append(subj)

                        elif "place" in node_type:
                            pref = node.get("prefLabel")
                            if isinstance(pref, list):
                                for p in pref:
                                    if isinstance(p, dict) and p.get("@value"):
                                        labels.append(str(p["@value"]))

                labels = list(dict.fromkeys(labels))[:8]
            except Exception:
                pass

        if title:
            text_parts.append(f"## {title}")
        if uri:
            text_parts.append(f"URI: {uri}")
        if created:
            text_parts.append(f"Created: {created}")
        if place:
            text_parts.append(f"Place: {place}")
        if labels:
            text_parts.append("Place labels: " + ", ".join(labels))
        if desc:
            text_parts.append(desc)
        if subjects:
            text_parts.append("Subjects: " + ", ".join(subjects[:10]))

        if len(text_parts) <= 1:
            text_parts.append(raw)

        text = "\n".join([p for p in text_parts if p]).strip()
        max_chars = max(200, int(self.valves.MAX_CHARS_PER_RECORD))
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "..."
        return text

    def _load_jsonl(self, source_path: str) -> List[Document]:
        if not os.path.exists(source_path):
            logger.warning(f"JSONL file not found: {source_path}")
            return []

        all_content: List[str] = []
        with open(source_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    text = self._extract_jsonl_record_text(item)
                    if text:
                        all_content.append(text)
                except json.JSONDecodeError:
                    continue

        if not all_content:
            return []

        # Group records so embedding count stays bounded.
        num_chunks = max(10, int(self.valves.TARGET_CHUNKS))
        lines_per_chunk = max(1, len(all_content) // num_chunks)

        documents: List[Document] = []
        for i in range(0, len(all_content), lines_per_chunk):
            chunk_lines = all_content[i : i + lines_per_chunk]
            combined_text = "\n\n".join(chunk_lines)
            documents.append(
                Document(
                    page_content=combined_text,
                    metadata={"source": "jsonl", "path": source_path, "chunk_id": len(documents)},
                )
            )

        logger.info(f"Loaded JSONL as {len(documents)} chunks (~{lines_per_chunk} records each)")
        return documents

    # ---------------------------
    # Markdown loading (improved)
    # ---------------------------
    def _load_markdown(self, source_path: str) -> List[Document]:
        """
        Chunk markdown primarily by headings.
        Goal: each '## item' section becomes its own doc (better retrieval).
        """
        if not os.path.exists(source_path):
            logger.warning(f"Markdown file not found: {source_path}")
            return []

        with open(source_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return []

        max_chars = max(800, int(self.valves.MAX_CHARS_PER_RECORD))  # smaller chunks
        # Prefer splitting on level-2 headings (your file uses "## ...")
        # Fallback to any heading if needed.
        sections = re.split(r"\n(?=##\s+)", "\n" + content)
        sections = [s.strip() for s in sections if s.strip()]

        if len(sections) <= 1:
            sections = re.split(r"\n(?=#+\s+)", "\n" + content)
            sections = [s.strip() for s in sections if s.strip()]

        docs: List[Document] = []
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue

            # If a single section is huge, split into smaller pieces
            if len(sec) > max_chars:
                # split by blank lines
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", sec) if p.strip()]
                cur = ""
                for p in paragraphs:
                    if len(cur) + len(p) + 2 <= max_chars:
                        cur = f"{cur}\n\n{p}".strip() if cur else p
                    else:
                        docs.append(Document(
                            page_content=cur,
                            metadata={"source": "markdown", "path": source_path, "chunk_id": len(docs)},
                        ))
                        cur = p
                if cur:
                    docs.append(Document(
                        page_content=cur,
                        metadata={"source": "markdown", "path": source_path, "chunk_id": len(docs)},
                    ))
            else:
                docs.append(Document(
                    page_content=sec,
                    metadata={"source": "markdown", "path": source_path, "chunk_id": len(docs)},
                ))

        logger.info(f"Loaded Markdown as {len(docs)} chunks from {source_path}")
        return docs

    # ---------------------------
    # Device selection
    # ---------------------------
    def _get_best_device(self, min_memory_gb: float = 2.0):
        import torch

        if not torch.cuda.is_available():
            return "cpu", {"device": "cpu", "trust_remote_code": True}

        torch.cuda.empty_cache()
        best_gpu = None
        best_free = 0.0

        for i in range(torch.cuda.device_count()):
            try:
                free_mem, _ = torch.cuda.mem_get_info(i)
                free_gb = free_mem / (1024**3)
                logger.info(f"GPU {i}: {free_gb:.2f} GB free")
                if free_gb > best_free:
                    best_free = free_gb
                    best_gpu = i
            except Exception:
                continue

        if best_gpu is not None and best_free >= float(min_memory_gb):
            return f"cuda:{best_gpu}", {"device": f"cuda:{best_gpu}", "trust_remote_code": True}

        logger.warning("Not enough GPU memory, using CPU")
        return "cpu", {"device": "cpu", "trust_remote_code": True}

    # ---------------------------
    # Cache save/load
    # ---------------------------
    def _save_cache(self, cache_path: Path, content_hash: str):
        try:
            if self.vectorstore:
                faiss_path = cache_path / f"faiss_{content_hash}"
                self.vectorstore.save_local(str(faiss_path))

            cache_data = {
                "documents": [(doc.page_content, doc.metadata) for doc in self.documents],
                "bm25_corpus": [self._bm25_tokenize(doc.page_content) for doc in self.documents],
                "embedding_model": self.valves.EMBEDDING_MODEL,
            }
            with open(cache_path / f"cache_{content_hash}.pkl", "wb") as f:
                pickle.dump(cache_data, f)

            logger.info(f"Cache saved to {cache_path}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _load_cache(self, cache_path: Path, content_hash: str) -> bool:
        try:
            faiss_path = cache_path / f"faiss_{content_hash}"
            pickle_path = cache_path / f"cache_{content_hash}.pkl"

            if not faiss_path.exists() or not pickle_path.exists():
                return False

            self.vectorstore = FAISS.load_local(
                str(faiss_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            with open(pickle_path, "rb") as f:
                cache_data = pickle.load(f)

            if cache_data.get("embedding_model") != self.valves.EMBEDDING_MODEL:
                return False

            self.documents = [Document(page_content=c, metadata=m) for c, m in cache_data["documents"]]
            self.bm25 = BM25Okapi(cache_data["bm25_corpus"])

            logger.info(f"Loaded {len(self.documents)} documents from cache")
            return True
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return False

    # ---------------------------
    # FAISS build
    # ---------------------------
    def _build_faiss_incremental(self, documents: List[Document]) -> FAISS:
        import torch

        model_config = self._get_model_config(self.valves.EMBEDDING_MODEL)
        batch_size = int(model_config.get("faiss_batch_size", 500))

        logger.info(f"Building FAISS index with {len(documents)} docs (batch_size={batch_size})")
        vectorstore = None
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            if batch_num % 10 == 1:
                logger.info(f"Batch {batch_num}/{total_batches}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, self.embeddings)
            else:
                # Avoid per-text embedding calls when possible by building a
                # batch index and merging it (faster for OpenAI embeddings).
                if hasattr(vectorstore, "merge_from"):
                    try:
                        tmp_store = FAISS.from_documents(batch, self.embeddings)
                        vectorstore.merge_from(tmp_store)
                    except Exception:
                        vectorstore.add_documents(batch)
                else:
                    vectorstore.add_documents(batch)

        return vectorstore

    # ---------------------------
    # Initialize
    # ---------------------------
    def _initialize(self):
        if self.initialized and self.current_embedding_model == self.valves.EMBEDDING_MODEL:
            return

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Initializing SearchCultureBot with {self.valves.EMBEDDING_MODEL}")
        start_time = time.time()

        model_config = self._get_model_config(self.valves.EMBEDDING_MODEL)
        provider = model_config.get("provider", "huggingface")

        if provider == "openai":
            logger.info(f"Using OpenAI embeddings: {self.valves.EMBEDDING_MODEL}")
            self.embeddings = OpenAIEmbeddingWrapper(
                client=self._get_openai_client(),
                model=self.valves.EMBEDDING_MODEL,
                batch_size=int(model_config.get("batch_size", 512)),
            )
        else:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            device, model_kwargs = self._get_best_device(model_config.get("min_gpu_memory_gb", 2.0))
            logger.info(f"Using HuggingFace embeddings on device: {device}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.valves.EMBEDDING_MODEL,
                model_kwargs=model_kwargs,
                encode_kwargs={"normalize_embeddings": True, "batch_size": model_config["batch_size"]},
            )
        self.current_embedding_model = self.valves.EMBEDDING_MODEL

        cache_path = self._get_cache_path()
        content_hash = self._get_content_hash()

        if self.valves.ENABLE_CACHE and self._load_cache(cache_path, content_hash):
            self.initialized = True
            logger.info(f"Initialized from cache in {time.time() - start_time:.2f}s")
            return

        source_path = self._get_source_path()
        source_format = self._detect_source_format(source_path)
        logger.info(f"Source: {source_path} ({source_format})")

        if source_format == "markdown":
            self.documents = self._load_markdown(source_path)
        else:
            self.documents = self._load_jsonl(source_path)

        if not self.documents:
            logger.warning("No documents loaded.")
            self.initialized = True
            return

        if self.valves.DEBUG:
            logger.info(f"Total docs loaded: {len(self.documents)}")
            logger.info(f"Sources present: {sorted(set(d.metadata.get('source') for d in self.documents))}")

        self.vectorstore = self._build_faiss_incremental(self.documents)

        logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi([self._bm25_tokenize(doc.page_content) for doc in self.documents])

        if self.valves.ENABLE_CACHE:
            self._save_cache(cache_path, content_hash)

        self.initialized = True
        logger.info(f"Initialized in {time.time() - start_time:.2f}s with {len(self.documents)} documents")

    # ---------------------------
    # Hybrid search
    # ---------------------------
    def _semantic_score_mode(self, semantic_results) -> str:
        """
        Decide how to interpret semantic scores.
        - distance: lower is better
        - similarity: higher is better
        - auto: if any score < 0 -> likely similarity; else distance
        """
        mode = (self.valves.SEMANTIC_SCORE_MODE or "auto").strip().lower()
        if mode in {"distance", "similarity"}:
            return mode
        # auto heuristic
        try:
            scores = [s for _, s in semantic_results]
            if any(s < 0 for s in scores):
                return "similarity"
        except Exception:
            pass
        return "distance"

    def _hybrid_search(self, query: str) -> List[Document]:
        if not self.documents:
            return []

        # Semantic search
        semantic_results = []
        if self.vectorstore:
            semantic_results = self.vectorstore.similarity_search_with_score(query, k=int(self.valves.TOP_K_SEMANTIC))

        # BM25 search
        bm25_results = []
        if self.bm25:
            q_tokens = self._bm25_tokenize(query)
            if q_tokens:
                scores = self.bm25.get_scores(q_tokens)
                top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: int(self.valves.TOP_K_BM25)]
                bm25_results = [(self.documents[i], float(scores[i])) for i in top_idx]

        doc_scores: Dict[int, float] = Counter()
        doc_map: Dict[int, Document] = {}

        # Normalize semantic scores robustly
        if semantic_results:
            mode = self._semantic_score_mode(semantic_results)
            sem_scores = [float(s) for _, s in semantic_results]
            s_min, s_max = min(sem_scores), max(sem_scores)
            denom = (s_max - s_min) if (s_max - s_min) != 0 else 1.0

            for doc, score in semantic_results:
                raw_key = doc.metadata.get("chunk_id")
                key = raw_key if isinstance(raw_key, int) else hash(doc.page_content)

                if mode == "distance":
                    # lower distance => higher normalized score
                    norm = (s_max - float(score)) / denom
                else:
                    # higher similarity => higher normalized score
                    norm = (float(score) - s_min) / denom

                doc_scores[key] = doc_scores.get(key, 0.0) + norm * float(self.valves.SEMANTIC_WEIGHT)
                doc_map[key] = doc

            if self.valves.DEBUG:
                logger.info(f"Semantic mode={mode}, scores(min={s_min:.4f}, max={s_max:.4f})")

        # Normalize BM25 scores
        if bm25_results:
            bm_scores = [float(s) for _, s in bm25_results]
            b_min, b_max = min(bm_scores), max(bm_scores)
            denom = (b_max - b_min) if (b_max - b_min) != 0 else 1.0

            for doc, score in bm25_results:
                raw_key = doc.metadata.get("chunk_id")
                key = raw_key if isinstance(raw_key, int) else hash(doc.page_content)
                norm = (float(score) - b_min) / denom  # higher BM25 => higher norm
                doc_scores[key] = doc_scores.get(key, 0.0) + norm * (1.0 - float(self.valves.SEMANTIC_WEIGHT))
                doc_map[key] = doc

        if not doc_scores:
            return []

        sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys[: int(self.valves.TOP_K_SEMANTIC) + int(self.valves.TOP_K_BM25)]]

    # ---------------------------
    # Context building
    # ---------------------------
    def _build_context(self, documents: List[Document]) -> str:
        """
        Build a structured context block from retrieved documents.
        Each chunk is prefixed with its index and any URI found in the content,
        so the model can reference them in its answer.
        """
        parts: List[str] = []
        total = 0
        max_len = int(self.valves.MAX_CONTEXT_LENGTH)

        for idx, doc in enumerate(documents, start=1):
            chunk_id = doc.metadata.get("chunk_id", "n/a")
            source_path = doc.metadata.get("path", "unknown")

            # Extract URI from the chunk text if present (for the header)
            uri_match = re.search(r"URI:\s*(https?://\S+)", doc.page_content)
            uri_hint = f" | URI: {uri_match.group(1)}" if uri_match else ""

            header = f"[Result {idx} | chunk_id={chunk_id}{uri_hint}]\n"

            remaining = max_len - total
            if remaining <= len(header):
                break

            content = doc.page_content.strip()
            allowed = remaining - len(header)

            if len(content) > allowed:
                content = content[:allowed].rsplit(" ", 1)[0] + "..."

            chunk_text = header + content
            parts.append(chunk_text)
            total += len(chunk_text)

            if total >= max_len:
                break

        return "\n\n---\n\n".join(parts)

    # ---------------------------
    # Generation (OpenAI ChatGPT)
    # ---------------------------
    def _generate_response(self, query: str, context: str) -> Generator[str, None, None]:
        has_context = bool(context.strip())

        if has_context:
            grounding_rules = (
                "2. **Grounding**: Whenever a retrieved result contains a URI that supports a claim, "
                "   cite it inline like this: ([source](URI)) — use the actual URI from the context.\n"
                "3. **Synthesis**: Do not list chunks one-by-one. Synthesise all relevant information "
                "   across multiple results into a single, coherent, well-structured answer.\n"
                "4. **Detail**: Provide thorough, analytical answers. Include dates, places, descriptions, "
                "   subjects, creators, and any other relevant metadata found in the context.\n"
                "5. **Completeness**: Use as many of the retrieved results as are relevant — do not "
                "   arbitrarily stop at a few results when more are available.\n"
                "6. **Knowledge fallback**: If the retrieved context is insufficient or silent on part of "
                "   the question, complement the answer with your own general knowledge — clearly marking "
                "   those parts as *(general knowledge)* so the user knows they are not from the dataset.\n"
            )
            context_block = (
                f"## Retrieved context ({context.count('---') + 1} results)\n\n"
                f"{context}\n\n"
                f"---\n\n"
            )
        else:
            # No retrieval results at all — answer purely from general knowledge
            grounding_rules = (
                "2. **Knowledge**: Answer using your general knowledge since no dataset results were found.\n"
                "3. **Transparency**: Mention briefly that no matching records were found in the dataset.\n"
            )
            context_block = "*(No matching records were retrieved from the dataset for this query.)*\n\n---\n\n"

        system_prompt = (
            "You are SearchCultureBot, an expert assistant for a cultural heritage dataset. "
            "Your primary source is the retrieved context passages below, but you should NEVER "
            "refuse to answer — always provide the most helpful response possible.\n\n"
            "## Rules\n"
            "1. **Language**: Detect the language of the user's question and respond exclusively in that language. If the question is in Greek, answer in Greek. If in English, answer in English. Never switch languages mid-answer.\n"
            + grounding_rules +
            "7. **Format**: Use Markdown headings, bullet lists, and bold text to organise long answers."
        )

        user_prompt = (
            context_block +
            f"## Question\n{query}"
        )

        user_prompt = (
            f"## Retrieved context ({context.count('---') + 1} results)\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"## Question\n{query}\n\n*(Answer in the same language as the question above)*"
        )

        try:
            client = self._get_openai_client()
            stream = client.chat.completions.create(
                model=self.valves.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=float(self.valves.TEMPERATURE),
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            yield f"\n\n⚠️ OpenAI API error: {e}"

    # ---------------------------
    # Pipeline entry
    # ---------------------------
    def pipe(
        self,
        user_message: str,
        model_id: str = None,
        messages: List[dict] = None,
        body: dict = None,
    ) -> Generator[str, None, None]:
        if not (self.initialized and self.current_embedding_model == self.valves.EMBEDDING_MODEL):
            yield "Initializing index (first run can take a while)...\n"
        self._initialize()

        if not self.documents:
            yield "No documents loaded. Check SOURCE_PATH/JSON_PATH and SOURCE_FORMAT. Also consider clearing cache."
            return

        if self.valves.DEBUG:
            yield f"[debug] docs={len(self.documents)} sources={sorted(set(d.metadata.get('source') for d in self.documents))}\n"

        relevant_docs = self._hybrid_search(user_message)
        if not relevant_docs:
            yield "No relevant documents found."
            return

        context = self._build_context(relevant_docs)
        yield from self._generate_response(user_message, context)
