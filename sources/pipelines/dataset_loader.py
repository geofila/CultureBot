"""
title: dataset_loader — filename-agnostic dataset discovery
author: SearchCultureBot
description: |
    Shared helper for both pipelines. Instead of requiring a fixed set of filenames
    (puretext_chunks.md, puretext_chunks.jsonl, searchculture_places_taxonomy.json…),
    the user drops whatever files they have into ./dataset — mounted at
    /app/pipelines/dataset — and this module works out what each one is:

        .md / .markdown  → text corpus for the RAG index (split on ## headings)
        .pdf             → text extracted per page, turned into "## <file> — page N"
                           sections so it chunks exactly like Markdown
        .jsonl / .ndjson → one record per line → {id → text} lookup for the KG path
        .json            → shape-sniffed: place taxonomy, filter vocabulary, or records
        anything else    → ignored (and reported in the startup log)

    Nothing is keyed on a filename; subfolders are walked too. Files that ship with the
    repo as format samples (*.example.*) and dataset/README.md are skipped, so a fresh
    clone does not index its own documentation.

    This module lives in the kg_jsons/ subdirectory of the deployed image because the
    OpenWebUI pipelines server imports every top-level .py in /app/pipelines as a
    Pipeline; helpers must stay out of that scan (same reason as sc_kg_nl2cypher.py).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ── What we recognise ─────────────────────────────────────────────────────────

MARKDOWN_EXTS = {".md", ".markdown"}
RECORD_EXTS = {".jsonl", ".ndjson"}
JSON_EXTS = {".json"}
PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = MARKDOWN_EXTS | RECORD_EXTS | JSON_EXTS | PDF_EXTS

# Directories never worth walking into.
SKIP_DIRS = {"cache", "__pycache__", ".git", ".ipynb_checkpoints", ".venv", "venv"}

# Files that are documentation or format samples, not data.
SKIP_FILENAMES = {"readme.md", "readme.markdown", ".gitkeep", ".ds_store", "thumbs.db"}
SKIP_SUBSTRINGS = (".example.",)

# Keys accepted when reading a record out of JSON/JSONL.
ID_KEYS = ("id", "uri", "url", "identifier")
TEXT_KEYS = ("text", "content", "chunk_text", "chunk", "body", "description")

# A JSON list whose objects carry any of these is a place taxonomy, not a record list.
PLACE_KEYS = ("path_text", "path", "label_el", "label_en", "alt_labels", "geonames_uri")

# Kinds that contribute nothing and are only reported.
INERT_KINDS = {"unsupported", "unreadable", "empty"}

_URI_LINE_RE = re.compile(
    r"^[ \t]*[-*]?[ \t]*(?:\*\*)?(?:URI|URL|ID|Identifier)(?:\*\*)?[ \t]*[:：][ \t]*<?(\S+?)>?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_URL_RE = re.compile(r"https?://\S+")
_H2_SPLIT_RE = re.compile(r"^##[ \t]+", re.MULTILINE)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class SourceFile:
    """One file found under the dataset folder, and what we made of it."""
    path: Path
    kind: str           # markdown | pdf | records | places | filters | unsupported | unreadable | empty
    detail: str = ""    # human-readable note for the startup log
    size: int = 0

    def __str__(self) -> str:
        note = f" — {self.detail}" if self.detail else ""
        return f"{self.path.name} [{self.kind}]{note}"


@dataclass
class Dataset:
    """Everything discovered under one dataset folder."""
    root: Path
    files: List[SourceFile] = field(default_factory=list)
    # (source label, markdown-ish text) for .md files and PDF-converted text
    texts: List[Tuple[str, str]] = field(default_factory=list)
    # {id → {"id": ..., "text": ...}} for the Knowledge-Graph lookup
    records: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # place taxonomy for the Cypher prompt (empty list when the user has none)
    places: List[Dict[str, Any]] = field(default_factory=list)
    # filter vocabulary, kept for completeness (no current reader)
    filters: Optional[Dict[str, Any]] = None

    # ── queries ───────────────────────────────────────────────────────────────

    @property
    def used_files(self) -> List[SourceFile]:
        return [f for f in self.files if f.kind not in INERT_KINDS]

    @property
    def is_empty(self) -> bool:
        return not self.texts and not self.records

    def rag_texts(self, mode: str = "auto") -> List[Tuple[str, str]]:
        """
        (source label, text) pairs to build the FAISS/BM25 index from.

        auto    — Markdown/PDF text if the user supplied any, otherwise the JSON records
                  (so a JSONL-only dataset is still searchable). This avoids indexing the
                  same corpus twice when .md and .jsonl are two views of the same records.
        text    — only Markdown/PDF
        records — only JSON/JSONL records
        all     — both, even if that duplicates content
        """
        mode = (mode or "auto").strip().lower()
        as_records = [(rid, rec["text"]) for rid, rec in self.records.items()]
        if mode == "text":
            return list(self.texts)
        if mode == "records":
            return as_records
        if mode == "all":
            return list(self.texts) + as_records
        return list(self.texts) if self.texts else as_records

    def fingerprint(self, fast: bool = False, extra: str = "") -> str:
        """
        Cache key over every file actually used: path + (content hash, or size+mtime when
        `fast`). Changing, adding or removing any dataset file changes this, so the index
        rebuilds by itself. Content is streamed, so a multi-GB JSONL is not read into RAM.
        """
        h = hashlib.md5()
        h.update(extra.encode("utf-8"))
        for src in sorted(self.used_files, key=lambda s: str(s.path)):
            h.update(str(src.path).encode("utf-8"))
            h.update(b"|")
            try:
                st = src.path.stat()
                if fast:
                    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
                    h.update(f"{st.st_size}|{mtime_ns}".encode("utf-8"))
                else:
                    with open(src.path, "rb") as fh:
                        for block in iter(lambda: fh.read(1024 * 1024), b""):
                            h.update(block)
            except OSError:
                h.update(b"<unreadable>")
            h.update(b"\n")
        return h.hexdigest()[:16]

    def summary(self) -> str:
        """One-line overview for the startup log."""
        if not self.files:
            return f"{self.root}: no files found"
        counts: Dict[str, int] = {}
        for src in self.files:
            counts[src.kind] = counts.get(src.kind, 0) + 1
        parts = ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items()))
        return (
            f"{self.root}: {parts} → "
            f"{len(self.texts)} text source(s), {len(self.records)} record(s), "
            f"{len(self.places)} place(s)"
        )

    def log_report(self, log=logger) -> None:
        """Log what was found, file by file, so a user can see their data was picked up."""
        log.info(f"Dataset scan — {self.summary()}")
        for src in self.files:
            level = log.info if src.kind not in INERT_KINDS else log.warning
            level(f"  • {src}")
        if self.is_empty:
            log.warning(
                f"No usable data found in {self.root}. Put your .md / .jsonl / .json / "
                f".pdf files in the dataset folder (any filename) and restart."
            )


# ── Readers ───────────────────────────────────────────────────────────────────

def _read_text(path: Path) -> str:
    # errors="replace" so one bad byte in a big corpus does not kill startup
    return path.read_text(encoding="utf-8", errors="replace")


def _record_from_obj(obj: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Pull (id, text) out of a dict using the accepted key spellings."""
    if not isinstance(obj, dict):
        return None
    rec_id = ""
    for key in ID_KEYS:
        value = obj.get(key)
        if value:
            rec_id = str(value).strip()
            break
    text = ""
    for key in TEXT_KEYS:
        value = obj.get(key)
        if value:
            text = str(value).strip()
            break
    if not text:
        return None
    return rec_id, text


def _read_jsonl(path: Path) -> Tuple[Dict[str, Dict[str, str]], int, int]:
    """Parse a .jsonl/.ndjson file. Returns (records, bad_lines, generated_ids)."""
    records: Dict[str, Dict[str, str]] = {}
    bad = 0
    generated = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            pair = _record_from_obj(obj)
            if pair is None:
                bad += 1
                continue
            rec_id, text = pair
            if not rec_id:
                # Keep the text anyway — it is still worth indexing for RAG. The synthetic
                # id simply never matches a graph result, which is the correct outcome.
                rec_id = f"{path.stem}#{line_num}"
                generated += 1
            records[rec_id] = {"id": rec_id, "text": text}
    return records, bad, generated


def _looks_like_places(data: Sequence[Any]) -> bool:
    for item in data[:20]:
        if isinstance(item, dict) and any(k in item for k in PLACE_KEYS):
            return True
    return False


def _read_json(path: Path) -> Tuple[str, Any, str]:
    """
    Sniff a .json file by shape. Returns (kind, payload, detail).

    list of place-ish dicts            → places
    list of record-ish dicts           → records
    dict with advanced_search_controls → filters
    dict of id → text | id → {text}    → records
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        if not data:
            # An empty list is the documented "I have no place taxonomy" placeholder.
            return "places", [], "empty list"
        if _looks_like_places(data):
            return "places", data, f"{len(data)} places"
        records: Dict[str, Dict[str, str]] = {}
        for i, obj in enumerate(data):
            pair = _record_from_obj(obj)
            if pair is None:
                continue
            rec_id, text = pair
            rec_id = rec_id or f"{path.stem}#{i}"
            records[rec_id] = {"id": rec_id, "text": text}
        if records:
            return "records", records, f"{len(records)} records"
        return "unsupported", None, "JSON list of unrecognised objects"

    if isinstance(data, dict):
        if "advanced_search_controls" in data or "source_html" in data:
            return "filters", data, "filter vocabulary"
        records = {}
        for key, value in data.items():
            if isinstance(value, str) and value.strip():
                records[str(key)] = {"id": str(key), "text": value.strip()}
            elif isinstance(value, dict):
                pair = _record_from_obj(value)
                if pair is not None:
                    records[str(key)] = {"id": str(key), "text": pair[1]}
        if records:
            return "records", records, f"{len(records)} records"
        return "unsupported", None, "JSON object of unrecognised shape"

    return "unsupported", None, "JSON is neither a list nor an object"


def _read_pdf(path: Path) -> Tuple[Optional[str], str]:
    """
    Extract text page by page and emit it as Markdown, one '## <file> — page N' section
    per page, so the same heading-based chunkers handle it with no special casing.
    """
    try:
        from pypdf import PdfReader  # installed by Dockerfile.pipelines
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError:
            return None, "no PDF library installed (pip install pypdf)"

    reader = PdfReader(str(path))
    sections: List[str] = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            sections.append(f"## {path.stem} — page {page_num}\n\n{text}")
    if not sections:
        return None, f"{len(reader.pages)} pages, no extractable text (scanned images?)"
    return "\n\n".join(sections), f"{len(sections)}/{len(reader.pages)} pages with text"


def harvest_markdown_records(text: str) -> Dict[str, Dict[str, str]]:
    """
    Build {uri → section text} from a Markdown corpus by reading the URI out of each
    '##' section. This is what lets the Knowledge-Graph path find an item's description
    when the user supplied Markdown only and no JSONL.
    """
    records: Dict[str, Dict[str, str]] = {}
    for raw in _H2_SPLIT_RE.split(text)[1:]:
        section = ("## " + raw).strip()
        match = _URI_LINE_RE.search(section)
        uri = match.group(1) if match else None
        if not uri:
            url_match = _URL_RE.search(section)
            uri = url_match.group(0).rstrip(".,;)") if url_match else None
        if uri and uri not in records:
            records[uri] = {"id": uri, "text": section}
    return records


# ── Discovery ─────────────────────────────────────────────────────────────────

def _should_skip(path: Path) -> bool:
    name = path.name.lower()
    if name.startswith("."):
        return True
    if name in SKIP_FILENAMES:
        return True
    return any(part in name for part in SKIP_SUBSTRINGS)


def iter_dataset_files(root: Path) -> List[Path]:
    """Every candidate file under root, sorted, with skip rules applied."""
    found: List[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if _should_skip(path):
            continue
        found.append(path)
    return found


def discover(
    root: "str | Path",
    extra_paths: Iterable["str | Path"] = (),
    harvest_ids_from_markdown: bool = True,
) -> Dataset:
    """
    Scan `root` (plus any explicitly given `extra_paths`) and return everything found.

    Never raises for missing or malformed data: an absent folder, an unreadable file or
    a JSON of an unexpected shape is recorded on the Dataset and logged, so the pipeline
    can start and tell the user what is wrong instead of crashing on import.
    """
    root = Path(root)
    dataset = Dataset(root=root)

    candidates: List[Path] = []
    if root.is_dir():
        candidates.extend(iter_dataset_files(root))
    else:
        logger.warning(f"Dataset folder not found: {root}")

    for extra in extra_paths:
        if not extra:
            continue
        extra_path = Path(extra)
        if extra_path.is_file() and extra_path not in candidates:
            candidates.append(extra_path)

    for path in candidates:
        ext = path.suffix.lower()
        try:
            size = path.stat().st_size
        except OSError:
            size = 0

        if ext not in SUPPORTED_EXTS:
            dataset.files.append(SourceFile(path, "unsupported", f"{ext or 'no extension'} not read", size))
            continue
        if size == 0:
            dataset.files.append(SourceFile(path, "empty", "0 bytes", size))
            continue

        try:
            if ext in MARKDOWN_EXTS:
                text = _read_text(path)
                if not text.strip():
                    dataset.files.append(SourceFile(path, "empty", "no text", size))
                    continue
                dataset.texts.append((path.name, text))
                detail = f"{len(text):,} chars"
                if harvest_ids_from_markdown:
                    harvested = harvest_markdown_records(text)
                    # Explicit JSON records win: they are the canonical text for an id.
                    for rec_id, rec in harvested.items():
                        dataset.records.setdefault(rec_id, rec)
                    if harvested:
                        detail += f", {len(harvested)} ids"
                dataset.files.append(SourceFile(path, "markdown", detail, size))

            elif ext in PDF_EXTS:
                text, detail = _read_pdf(path)
                if text is None:
                    dataset.files.append(SourceFile(path, "unreadable", detail, size))
                    continue
                dataset.texts.append((path.name, text))
                dataset.files.append(SourceFile(path, "pdf", detail, size))

            elif ext in RECORD_EXTS:
                records, bad, generated = _read_jsonl(path)
                if not records:
                    dataset.files.append(SourceFile(path, "unreadable", "no usable records", size))
                    continue
                dataset.records.update(records)
                detail = f"{len(records):,} records"
                if generated:
                    detail += f", {generated} without an id"
                if bad:
                    detail += f", {bad} lines skipped"
                dataset.files.append(SourceFile(path, "records", detail, size))

            else:  # .json
                kind, payload, detail = _read_json(path)
                if kind == "places":
                    dataset.places.extend(payload)
                elif kind == "records":
                    dataset.records.update(payload)
                elif kind == "filters":
                    dataset.filters = payload
                dataset.files.append(SourceFile(path, kind, detail, size))

        except Exception as exc:  # malformed file must not stop the whole pipeline
            dataset.files.append(SourceFile(path, "unreadable", f"{type(exc).__name__}: {exc}", size))

    return dataset
