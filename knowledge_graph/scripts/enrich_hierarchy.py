"""
enrich_hierarchy.py
===================

Build the FULL hierarchical trees that ingestion leaves shallow.

Problem it solves
-----------------
`ingest.py` only creates edges declared inside each JSONL record. A place
embeds its own `isPartOf` (one level up), but the parent appears only as a
bare target id with no `isPartOf` of its own, so the chain stops one hop up.
Concepts have the same issue with `broader` / `broadMatch`.

What it does
------------
Every authority node has a dereferenceable `semantics.gr` URI as its `id`.
This script walks those URIs *upward*, following:

    dcterms:isPartOf   (places)   -> (child)-[:IS_PART_OF]->(parent)
    skos:broader       (concepts) -> (child)-[:BROADER_THAN]->(parent)
    skos:broadMatch    (concepts) -> (child)-[:BROAD_MATCH]->(parent)

It recurses to the roots (country / continent / top concept) with a global
visited set, an on-disk cache, rate limiting and retries. Everything is MERGEd,
so it can be re-run or resumed safely. Edge names/directions match ingest.py.

Response formats handled
------------------------
A single-URI GET on semantics.gr returns EXPANDED JSON-LD:

    [ { "@id": "...", "http://www.w3.org/2004/02/skos/core#broader": [ {"@id": "..."} ], ... } ]

Some deployments / content-negotiation return the site's internal ENVELOPE
instead (a dict keyed by full URI, or a single object carrying
`propertyValues` with `skos:broader.value[].resourceURI`, relative paths).
Both are parsed. If a fetch fails, the reason is recorded so a --dry-run makes
the real cause obvious instead of silently reporting 0 edges.

Requirements
------------
    pip install neo4j requests python-dotenv tqdm

Usage
-----
    python knowledge_graph/scripts/enrich_hierarchy.py --cache knowledge_graph/data/authority_cache.json
    python knowledge_graph/scripts/enrich_hierarchy.py --dry-run --max-nodes 50 --debug --cache knowledge_graph/data/authority_cache.json
    python knowledge_graph/scripts/enrich_hierarchy.py --only-frontier --sleep 0.2 --cache knowledge_graph/data/authority_cache.json

Neo4j connection settings can be supplied through the repository .env file.
"""

import argparse
import json
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import requests
from neo4j import GraphDatabase
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

AUTHORITY_PREFIX = "http://semantics.gr/authorities/"

# ---- expanded JSON-LD predicate URIs -----------------------------------
P_ISPARTOF   = "http://purl.org/dc/terms/isPartOf"
P_BROADER    = "http://www.w3.org/2004/02/skos/core#broader"
P_BROADMATCH = "http://www.w3.org/2004/02/skos/core#broadMatch"
P_PREFLABEL  = "http://www.w3.org/2004/02/skos/core#prefLabel"
P_ALTLABEL   = "http://www.w3.org/2004/02/skos/core#altLabel"
P_TYPE       = "@type"
P_LAT        = "http://www.w3.org/2003/01/geo/wgs84_pos#lat"
P_LONG       = "http://www.w3.org/2003/01/geo/wgs84_pos#long"
P_FEATCLASS  = "https://www.semantics.gr/authorities/schemanamespaces/ekt#featureClass"

# predicate -> (relationship name, fallback parent label). Direction is always
# (child)-[REL]->(parent), matching ingest.py.
UPWARD_PREDICATES = {
    P_ISPARTOF:   ("IS_PART_OF",   "Place"),
    P_BROADER:    ("BROADER_THAN", "Concept"),
    P_BROADMATCH: ("BROAD_MATCH",  "Concept"),
}

# ---- envelope (internal API) property keys -----------------------------
ENVELOPE_UPWARD = {
    "dcterms:isPartOf": ("IS_PART_OF",   "Place"),
    "skos:broader":     ("BROADER_THAN", "Concept"),
    "skos:broadMatch":  ("BROAD_MATCH",  "Concept"),
}


# ============================================================
# JSON-LD helpers (expanded form: full-URI keys, list values)
# ============================================================

def _ids(obj: Dict[str, Any], key: str) -> List[str]:
    out: List[str] = []
    for v in obj.get(key, []) or []:
        if isinstance(v, dict) and isinstance(v.get("@id"), str):
            out.append(v["@id"])
        elif isinstance(v, str) and v.startswith("http"):
            out.append(v)
    return out


def _lang(obj: Dict[str, Any], key: str, lang: str) -> Optional[str]:
    for v in obj.get(key, []) or []:
        if isinstance(v, dict) and v.get("@language") == lang and "@value" in v:
            return str(v["@value"]).strip()
    return None


def _literals(obj: Dict[str, Any], key: str) -> List[str]:
    out: List[str] = []
    for v in obj.get(key, []) or []:
        if isinstance(v, dict) and "@value" in v:
            s = str(v["@value"]).strip()
            if s:
                out.append(s)
    return out


def _first_value(obj: Dict[str, Any], key: str) -> Optional[str]:
    for v in obj.get(key, []) or []:
        if isinstance(v, dict) and "@value" in v:
            return str(v["@value"]).strip()
    return None


def _type_label_jsonld(obj: Dict[str, Any]) -> Optional[str]:
    types = obj.get(P_TYPE) or []
    if isinstance(types, str):
        types = [types]
    for t in types:
        if isinstance(t, str):
            return t.rsplit("/", 1)[-1].rsplit("#", 1)[-1]  # edm Place / skos Concept
    return None


# ============================================================
# Format detection + entry selection
# ============================================================

def is_envelope(entry: Dict[str, Any]) -> bool:
    return isinstance(entry, dict) and "propertyValues" in entry


def _abs_uri(ref: Optional[str]) -> Optional[str]:
    if not ref:
        return None
    ref = ref.strip()
    if ref.startswith("http://") or ref.startswith("https://"):
        return ref
    return AUTHORITY_PREFIX + ref.lstrip("/")


def _env_first(pv: Dict[str, Any], key: str) -> Optional[str]:
    """First literal (or resourceURI) value under an envelope propertyValues key."""
    block = pv.get(key) or {}
    for it in (block.get("value") or []):
        if isinstance(it, dict):
            val = it.get("value") or it.get("resourceURI")
            if val:
                return str(val).strip()
    return None


def select_entry(payload: Any, node_id: str) -> Optional[Dict[str, Any]]:
    """Pull the record for node_id out of whatever the server returned."""
    if payload is None:
        return None

    # expanded JSON-LD: a list of node objects
    if isinstance(payload, list):
        for e in payload:
            if isinstance(e, dict) and e.get("@id") == node_id:
                return e
        return payload[0] if payload and isinstance(payload[0], dict) else None

    if isinstance(payload, dict):
        # JSON-LD with @graph
        graph = payload.get("@graph")
        if isinstance(graph, list) and graph:
            for e in graph:
                if isinstance(e, dict) and e.get("@id") == node_id:
                    return e
            return graph[0]
        # envelope keyed by full URI
        if isinstance(payload.get(node_id), dict):
            return payload[node_id]
        # single envelope object
        if "propertyValues" in payload:
            return payload
        # envelope dict keyed by *some* URIs (match by rdfAbout suffix)
        dict_vals = [v for v in payload.values() if isinstance(v, dict)]
        if dict_vals and any("propertyValues" in v for v in dict_vals):
            for k, v in payload.items():
                if k == node_id:
                    return v
                ra = v.get("rdfAbout") if isinstance(v, dict) else None
                if isinstance(ra, str) and node_id.endswith(ra):
                    return v
            return dict_vals[0]
        # otherwise assume it's a single expanded JSON-LD object
        return payload
    return None


# ============================================================
# Unified extraction (works on JSON-LD or envelope entries)
# ============================================================

def extract_parents(entry: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Return [(rel_name, parent_label, parent_id)] climbing upward."""
    out: List[Tuple[str, str, str]] = []
    if is_envelope(entry):
        pv = entry.get("propertyValues") or {}
        for key, (rel, lbl) in ENVELOPE_UPWARD.items():
            block = pv.get(key) or {}
            for item in (block.get("value") or []):
                if not isinstance(item, dict):
                    continue
                ref = item.get("resourceURI") or item.get("@id")
                if item.get("resource", True) and ref:
                    pid = _abs_uri(ref)
                    if pid:
                        out.append((rel, lbl, pid))
    else:
        for pred, (rel, lbl) in UPWARD_PREDICATES.items():
            for pid in _ids(entry, pred):
                out.append((rel, lbl, pid))
    # de-dup while preserving nothing in particular
    return list({t for t in out})


def extract_meta(entry: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """Label / type / coords for upserting the node itself."""
    if is_envelope(entry):
        owl = entry.get("owlClassQName") or ""
        type_label = owl.split(":")[-1] if owl else (
            (entry.get("contextualFamily") or "Unknown").title()
        )
        pref = entry.get("prefLabel") or {}
        pref_en = pref.get("EN") or pref.get("en")
        pref_el = pref.get("EL") or pref.get("el")
        alts = [a.get("value") for a in (entry.get("altLabel") or [])
                if isinstance(a, dict) and a.get("value")]
        label = pref_en or pref_el or entry.get("prefLabelInPrefLang") or node_id
        pv = entry.get("propertyValues") or {}
        return dict(type_label=type_label or "Unknown", pref_en=pref_en,
                    pref_el=pref_el, alts=alts,
                    lat=_env_first(pv, "wgs84_pos:lat"),
                    long=_env_first(pv, "wgs84_pos:long"),
                    feat=_env_first(pv, "ekt:featureClass"), label=label)
    # JSON-LD
    type_label = _type_label_jsonld(entry) or "Unknown"
    pref_en = _lang(entry, P_PREFLABEL, "en")
    pref_el = _lang(entry, P_PREFLABEL, "el")
    label = pref_en or pref_el or _first_value(entry, P_PREFLABEL) or node_id
    feat = _ids(entry, P_FEATCLASS)
    return dict(type_label=type_label, pref_en=pref_en, pref_el=pref_el,
                alts=_literals(entry, P_ALTLABEL),
                lat=_first_value(entry, P_LAT), long=_first_value(entry, P_LONG),
                feat=(feat[0] if feat else None), label=label)


# ============================================================
# Disk cache
# ============================================================

class Cache:
    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def get(self, key: str):
        return self.data.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def flush(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)


# ============================================================
# Fetch one authority document (records failure reason)
# ============================================================

def fetch_authority(uri: str, session: requests.Session, timeout: float,
                    retries: int = 3, backoff: float = 1.5
                    ) -> Tuple[Optional[Any], Optional[str]]:
    """Return (payload, error). payload is the raw decoded JSON (list or dict)."""
    headers = {"Accept": "application/ld+json, application/json;q=0.9, */*;q=0.1"}
    last_err = None
    for attempt in range(retries):
        try:
            resp = session.get(uri, headers=headers, timeout=timeout,
                               allow_redirects=True)
            if resp.status_code == 404:
                return None, "http 404"
            resp.raise_for_status()
            try:
                return resp.json(), None
            except ValueError:
                return None, "non-json body (ctype=%s, first80=%r)" % (
                    resp.headers.get("content-type"), resp.text[:80])
        except Exception as e:  # noqa: BLE001
            last_err = "%s: %s" % (type(e).__name__, e)
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    return None, last_err


# ============================================================
# Graph writes
# ============================================================

def upsert_node(tx, node_id: str, meta: Dict[str, Any]) -> None:
    tx.run(
        """
        MERGE (n:Entity {id: $id})
        SET n.label = $label,
            n.pref_label_en = $pref_en,
            n.pref_label_el = $pref_el,
            n.alt_labels = $alts,
            n.lat = $lat,
            n.long = $long,
            n.feature_class = $feat
        SET n:$($type_label)
        """,
        id=node_id, label=meta["label"], pref_en=meta["pref_en"],
        pref_el=meta["pref_el"], alts=meta["alts"], lat=meta["lat"],
        long=meta["long"], feat=meta["feat"], type_label=meta["type_label"],
    )


def link_parent(tx, child_id: str, parent_id: str, rel_name: str,
                parent_label: str) -> None:
    tx.run(
        f"""
        MATCH (child:Entity {{id: $child_id}})
        MERGE (parent:Entity {{id: $parent_id}})
        ON CREATE SET parent.label = $parent_id
        SET parent:$($parent_label)
        MERGE (child)-[:{rel_name}]->(parent)
        """,
        child_id=child_id, parent_id=parent_id, parent_label=parent_label,
    )


# ============================================================
# Seed selection
# ============================================================

def get_seed_ids(session, only_frontier: bool) -> List[str]:
    if only_frontier:
        query = """
        MATCH (n:Entity)
        WHERE n.id STARTS WITH $prefix
          AND NOT (n)-[:IS_PART_OF|BROADER_THAN|BROAD_MATCH]->()
        RETURN n.id AS id
        """
    else:
        query = """
        MATCH (n:Entity)
        WHERE n.id STARTS WITH $prefix
        RETURN n.id AS id
        """
    return [r["id"] for r in session.run(query, prefix=AUTHORITY_PREFIX)]


# ============================================================
# Main BFS enrichment
# ============================================================

def enrich(args) -> None:
    password = args.password or os.getenv("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("Set NEO4J_PASSWORD or pass --password.")

    cache = Cache(args.cache)
    http = requests.Session()
    http.headers.update({"User-Agent": "culturebot-enrich/1.1"})

    stats = dict(fetched_ok=0, notfound_404=0, transport_fail=0, with_parents=0,
                 without_parents=0, edges=0)
    sample_404: List[str] = []
    sample_errors: List[str] = []
    shown_debug = {"done": False}

    with GraphDatabase.driver(args.uri, auth=(args.user, password)) as driver:
        with driver.session() as session:
            print("Selecting seed authority nodes...")
            seeds = get_seed_ids(session, args.only_frontier)
            print(f"  {len(seeds)} seed nodes.")

            visited = set()
            queue = deque(seeds)
            pbar = tqdm(total=len(seeds), desc="Enriching hierarchy", unit="node")

            while queue:
                node_id = queue.popleft()
                if node_id in visited:
                    continue
                visited.add(node_id)
                pbar.update(1)

                if args.max_nodes and (stats["fetched_ok"] + stats["notfound_404"] + stats["transport_fail"]) >= args.max_nodes:
                    break
                if not node_id.startswith(AUTHORITY_PREFIX):
                    continue

                # fetch (cache-first)
                if node_id in cache:
                    payload = cache.get(node_id)
                    err = None if payload is not None else "cached-miss"
                else:
                    payload, err = fetch_authority(node_id, http, args.timeout)
                    cache.set(node_id, payload)  # may be None (cached failure)
                    if args.sleep:
                        time.sleep(args.sleep)
                    if (stats["fetched_ok"] + stats["notfound_404"] + stats["transport_fail"]) % 200 == 0:
                        cache.flush()

                if payload is None:
                    if err and "404" in err:
                        stats["notfound_404"] += 1
                        if len(sample_404) < 5:
                            sample_404.append(node_id)
                    else:
                        stats["transport_fail"] += 1
                        if err and len(sample_errors) < 5:
                            sample_errors.append(f"{node_id}  ->  {err}")
                    continue
                stats["fetched_ok"] += 1

                entry = select_entry(payload, node_id)
                if not entry:
                    stats["without_parents"] += 1
                    continue

                if args.debug and not shown_debug["done"]:
                    fmt = "envelope" if is_envelope(entry) else "json-ld"
                    keys = list(entry.keys())[:8]
                    tqdm.write(f"[debug] first ok fetch: format={fmt} "
                               f"keys={keys}")
                    shown_debug["done"] = True

                parents = extract_parents(entry)
                if parents:
                    stats["with_parents"] += 1
                else:
                    stats["without_parents"] += 1

                if not args.dry_run:
                    meta = extract_meta(entry, node_id)
                    session.execute_write(upsert_node, node_id, meta)

                for rel_name, parent_label, parent_id in parents:
                    if not args.dry_run:
                        session.execute_write(
                            link_parent, node_id, parent_id, rel_name, parent_label
                        )
                    stats["edges"] += 1
                    if parent_id not in visited:
                        queue.append(parent_id)
                        pbar.total += 1
                        pbar.refresh()

            pbar.close()
            cache.flush()

    print("\n=== summary ===")
    print(f"  fetched ok         : {stats['fetched_ok']}")
    print(f"  not published (404): {stats['notfound_404']}")
    print(f"  transport FAILED   : {stats['transport_fail']}")
    print(f"  nodes w/ parents   : {stats['with_parents']}")
    print(f"  nodes w/o parents  : {stats['without_parents']}")
    print(f"  upward edges {'(planned)' if args.dry_run else '(written)'} : {stats['edges']}")
    if stats["notfound_404"]:
        print("  404s (benign - those authorities are not dereferenceable,")
        print("        so they stay as leaf nodes). e.g.:")
        for i in sample_404:
            print("    -", i)
    if stats["transport_fail"]:
        print("  transport errors (ACTIONABLE - proxy / TLS / DNS / outbound):")
        for e in sample_errors:
            print("    -", e)
    elif stats["edges"] == 0 and stats["fetched_ok"] > 0:
        print("  -> fetches succeed but no parents parsed; re-run with --debug")
        print("     to print the response shape.")
    print(f"Cache saved to {args.cache}")


def parse_args():
    p = argparse.ArgumentParser(description="Build full authority hierarchies in the CULTUREBOT graph.")
    p.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    p.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    p.add_argument("--password", default=None, help="defaults to $NEO4J_PASSWORD")
    p.add_argument("--cache", default="authority_cache.json")
    p.add_argument("--sleep", type=float, default=0.1, help="seconds between HTTP fetches")
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--only-frontier", action="store_true",
                   help="start only from authority nodes with no upward edge yet")
    p.add_argument("--max-nodes", type=int, default=0, help="cap HTTP fetches (0 = unlimited)")
    p.add_argument("--dry-run", action="store_true", help="fetch + report, write nothing")
    p.add_argument("--debug", action="store_true", help="print the shape of the first fetched doc")
    return p.parse_args()


if __name__ == "__main__":
    enrich(parse_args())
