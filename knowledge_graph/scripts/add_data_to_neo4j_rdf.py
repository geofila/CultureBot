#!/usr/bin/env python3
"""
Robust JSONL -> Neo4j RDF importer using neosemantics (n10s).

Assumptions:
- Each line in the input JSONL file is one complete JSON-LD document.
- Neo4j has the n10s plugin installed and available.
- The target DB is intended for RDF import via n10s.

Features:
- Progress bar with tqdm
- Checkpointing (resume after crash)
- Batch import for speed
- Per-record fallback when a batch fails
- Failed-record log
- Ensures required n10s URI constraint exists
- Best-effort n10s graph config initialization

Usage example:
Set NEO4J_PASSWORD in the environment, then run:
python knowledge_graph/scripts/add_data_to_neo4j_rdf.py \
  --uri bolt://localhost:7687 \
  --user neo4j \
  --input knowledge_graph/data/edm.jsonl \
  --checkpoint knowledge_graph/data/rdf-checkpoint.json \
  --failed-log knowledge_graph/data/rdf-failed-lines.jsonl \
  --batch-size 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, DriverError
from tqdm import tqdm


# ----------------------------
# Data models
# ----------------------------

@dataclass
class ImportRow:
    line_number: int
    payload: str


# ----------------------------
# File / checkpoint helpers
# ----------------------------

def count_lines(filepath: Path) -> int:
    with filepath.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_checkpoint(path: Path) -> int:
    """
    Returns the next line number to process (1-based).
    If no checkpoint exists, start at line 1.
    """
    if not path.exists():
        return 1
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("next_line", 1))


def save_checkpoint(path: Path, next_line: int) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump({"next_line": next_line, "updated_at": time.time()}, f, indent=2)
    tmp.replace(path)


def append_failed_line(path: Path, line_number: int, payload: str, error: str) -> None:
    record = {
        "line_number": line_number,
        "error": error,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def batched_rows(
    filepath: Path,
    start_line: int,
    batch_size: int,
) -> Iterable[List[ImportRow]]:
    """
    Yields batches of ImportRow, starting from 1-based line number start_line.
    """
    batch: List[ImportRow] = []
    with filepath.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if lineno < start_line:
                continue

            payload = line.strip()
            if not payload:
                continue

            batch.append(ImportRow(line_number=lineno, payload=payload))
            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


# ----------------------------
# Neo4j helpers
# ----------------------------

def ensure_n10s_available(driver) -> None:
    query = """
    SHOW PROCEDURES
    YIELD name
    WHERE name = 'n10s.rdf.import.inline'
    RETURN count(*) AS cnt
    """
    with driver.session() as session:
        cnt = session.run(query).single()["cnt"]
        if cnt == 0:
            raise RuntimeError(
                "n10s.rdf.import.inline not found. "
                "Make sure you are connected to the RDF Neo4j instance with the n10s plugin loaded."
            )


def ensure_constraint(driver) -> None:
    """
    Required by n10s RDF import.
    """
    query = """
    CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS
    FOR (r:Resource)
    REQUIRE r.uri IS UNIQUE
    """
    with driver.session() as session:
        session.run(query).consume()


def get_graph_config(driver) -> dict | None:
    """
    Return current n10s graph config as a dict, or None if not initialized.
    """
    query = """
    CALL n10s.graphconfig.show()
    YIELD param, value
    RETURN collect({param: param, value: value}) AS items
    """
    with driver.session() as session:
        try:
            rec = session.run(query).single()
        except Neo4jError as e:
            msg = str(e).lower()
            if "graph config" in msg and ("not found" in msg or "not initialized" in msg):
                return None
            raise

        if rec is None:
            return None

        items = rec["items"] or []
        if not items:
            return None

        return {item["param"]: item["value"] for item in items}


def ensure_graph_config(driver, desired: dict[str, object] | None = None) -> dict:
    """
    Ensure n10s graph config exists.
    - If missing: initialize with desired config (or defaults if desired is None)
    - If present: return existing config unchanged
    """
    existing = get_graph_config(driver)
    if existing is not None:
        print("n10s graph config already exists. Reusing existing configuration.")
        return existing

    desired = desired or {"handleVocabUris": "SHORTEN", "handleMultival": "ARRAY"}

    query = "CALL n10s.graphconfig.init($cfg)"
    with driver.session() as session:
        session.run(query, cfg=desired).consume()

    print(f"Initialized n10s graph config: {desired}")
    return desired


def import_batch(driver, rows: List[ImportRow]) -> Dict[str, Any]:
    """
    Imports a batch of JSON-LD payloads using n10s.rdf.import.inline.
    Returns aggregate stats.

    If any row causes the batch query to fail, the caller should catch the exception
    and fall back to row-by-row import.
    """
    cypher = """
    UNWIND $rows AS row
    CALL {
      WITH row
      CALL n10s.rdf.import.inline(row.payload, "JSON-LD")
      YIELD terminationStatus, triplesLoaded, triplesParsed, namespaces, extraInfo, callParams
      RETURN
        row.line_number AS line_number,
        terminationStatus AS terminationStatus,
        triplesLoaded AS triplesLoaded,
        triplesParsed AS triplesParsed,
        extraInfo AS extraInfo
    }
    RETURN
      count(*) AS rowsProcessed,
      sum(coalesce(triplesLoaded, 0)) AS triplesLoaded,
      sum(coalesce(triplesParsed, 0)) AS triplesParsed,
      collect({
        line_number: line_number,
        terminationStatus: terminationStatus,
        triplesLoaded: triplesLoaded,
        triplesParsed: triplesParsed,
        extraInfo: extraInfo
      }) AS details
    """
    payload = {
        "rows": [
            {"line_number": r.line_number, "payload": r.payload}
            for r in rows
        ]
    }

    with driver.session() as session:
        rec = session.run(cypher, payload).single()
        if rec is None:
            raise RuntimeError("Batch import returned no result.")
        return {
            "rowsProcessed": rec["rowsProcessed"],
            "triplesLoaded": rec["triplesLoaded"],
            "triplesParsed": rec["triplesParsed"],
            "details": rec["details"],
        }


def import_single(driver, row: ImportRow) -> Dict[str, Any]:
    cypher = """
    CALL n10s.rdf.import.inline($payload, "JSON-LD")
    YIELD terminationStatus, triplesLoaded, triplesParsed, namespaces, extraInfo, callParams
    RETURN terminationStatus, triplesLoaded, triplesParsed, extraInfo
    """
    with driver.session() as session:
        rec = session.run(cypher, payload=row.payload).single()
        if rec is None:
            raise RuntimeError(f"Line {row.line_number}: no result returned")
        return {
            "terminationStatus": rec["terminationStatus"],
            "triplesLoaded": rec["triplesLoaded"],
            "triplesParsed": rec["triplesParsed"],
            "extraInfo": rec["extraInfo"],
        }


# ----------------------------
# Main import logic
# ----------------------------

def process_file(
    driver,
    input_path: Path,
    checkpoint_path: Path,
    failed_log_path: Path,
    batch_size: int,
    max_retries: int,
    retry_sleep: float,
) -> None:
    total_lines = count_lines(input_path)
    next_line = load_checkpoint(checkpoint_path)

    if next_line > total_lines:
        print(f"Checkpoint says next_line={next_line}, but file has only {total_lines} lines.")
        print("Nothing to do.")
        return

    print(f"Input file      : {input_path}")
    print(f"Total lines     : {total_lines}")
    print(f"Starting at line: {next_line}")
    print(f"Batch size      : {batch_size}")
    print(f"Checkpoint file : {checkpoint_path}")
    print(f"Failed log      : {failed_log_path}")

    progress_total = total_lines - next_line + 1
    pbar = tqdm(total=progress_total, desc="Importing RDF JSONL", unit="line")

    if next_line > 1:
        pbar.update(next_line - 1 - (next_line - 1))  # no-op, keeps logic explicit

    last_committed_line = next_line - 1
    total_triples_loaded = 0
    total_triples_parsed = 0
    bad_lines = 0

    for batch in batched_rows(input_path, start_line=next_line, batch_size=batch_size):
        batch_start = batch[0].line_number
        batch_end = batch[-1].line_number

        # Retry the batch first
        batch_success = False
        batch_error = None

        for attempt in range(1, max_retries + 1):
            try:
                result = import_batch(driver, batch)
                total_triples_loaded += int(result["triplesLoaded"] or 0)
                total_triples_parsed += int(result["triplesParsed"] or 0)

                # Inspect individual statuses
                bad_in_batch = []
                for d in result["details"]:
                    if d["terminationStatus"] != "OK":
                        bad_in_batch.append(d)

                if bad_in_batch:
                    # Fallback to per-row handling for those lines
                    raise RuntimeError(
                        f"Batch {batch_start}-{batch_end} returned non-OK statuses for "
                        f"{len(bad_in_batch)} row(s). Falling back to per-line import."
                    )

                batch_success = True
                break

            except Exception as e:
                batch_error = e
                if attempt < max_retries:
                    time.sleep(retry_sleep)

        if batch_success:
            last_committed_line = batch_end
            save_checkpoint(checkpoint_path, last_committed_line + 1)
            pbar.update(len(batch))
            pbar.set_postfix(
                triples_loaded=total_triples_loaded,
                bad_lines=bad_lines,
                last_line=last_committed_line,
            )
            continue

        # Batch failed after retries -> fallback to line-by-line
        print(
            f"\nBatch {batch_start}-{batch_end} failed after {max_retries} attempt(s). "
            f"Falling back to single-line processing.\nLast batch error: {batch_error}"
        )

        for row in batch:
            row_success = False
            row_error = None

            for attempt in range(1, max_retries + 1):
                try:
                    result = import_single(driver, row)
                    if result["terminationStatus"] != "OK":
                        raise RuntimeError(
                            f"terminationStatus={result['terminationStatus']}, "
                            f"extraInfo={result['extraInfo']}"
                        )

                    total_triples_loaded += int(result["triplesLoaded"] or 0)
                    total_triples_parsed += int(result["triplesParsed"] or 0)

                    row_success = True
                    break

                except Exception as e:
                    row_error = e
                    if attempt < max_retries:
                        time.sleep(retry_sleep)

            if not row_success:
                bad_lines += 1
                append_failed_line(
                    failed_log_path,
                    line_number=row.line_number,
                    payload=row.payload,
                    error=str(row_error),
                )
                print(f"Skipped bad line {row.line_number}: {row_error}")

            last_committed_line = row.line_number
            save_checkpoint(checkpoint_path, last_committed_line + 1)
            pbar.update(1)
            pbar.set_postfix(
                triples_loaded=total_triples_loaded,
                bad_lines=bad_lines,
                last_line=last_committed_line,
            )

    pbar.close()

    print("\nDone.")
    print(f"Last committed line : {last_committed_line}")
    print(f"Total triples loaded: {total_triples_loaded}")
    print(f"Total triples parsed: {total_triples_parsed}")
    print(f"Bad lines           : {bad_lines}")
    print(f"Checkpoint saved to : {checkpoint_path}")
    print(f"Failed lines logged : {failed_log_path}")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import JSONL JSON-LD records into Neo4j via n10s.")
    parser.add_argument("--uri", required=True, help="Neo4j Bolt URI, e.g. bolt://localhost:7688")
    parser.add_argument("--user", required=True, help="Neo4j username")
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"), help="Neo4j password (defaults to NEO4J_PASSWORD)")
    parser.add_argument("--input", required=True, help="Path to JSONL input file")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint JSON file")
    parser.add_argument("--failed-log", required=True, help="Path to failed-lines JSONL log")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of lines per batch")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for batch or row")
    parser.add_argument("--retry-sleep", type=float, default=2.0, help="Seconds between retries")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.password:
        print("Set NEO4J_PASSWORD or pass --password.", file=sys.stderr)
        return 1

    input_path = Path(args.input).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    failed_log_path = Path(args.failed_log).expanduser().resolve()

    if not input_path.exists():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    failed_log_path.parent.mkdir(parents=True, exist_ok=True)

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    try:
        ensure_n10s_available(driver)
        ensure_constraint(driver)
        ensure_graph_config(driver)

        process_file(
            driver=driver,
            input_path=input_path,
            checkpoint_path=checkpoint_path,
            failed_log_path=failed_log_path,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
        )
    except (RuntimeError, DriverError, Neo4jError) as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        return 2
    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
