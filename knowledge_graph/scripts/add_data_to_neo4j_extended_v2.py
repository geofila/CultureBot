import itertools
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Utility helpers
# ============================================================

def _safe_label(label: str) -> str:
    """Sanitize dynamic Neo4j labels to avoid invalid characters."""
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", label or "")
    if not cleaned:
        return "Unknown"
    if cleaned[0].isdigit():
        return f"T_{cleaned}"
    return cleaned


def _type_suffix(type_field: Any) -> str:
    """
    Return the JSON-LD @type suffix used as a Neo4j label.

    Examples:
      - "skos:Concept" -> "Concept"
      - ["edm:ProvidedCHO", "Other"] -> "ProvidedCHO"
    """
    if not type_field:
        return "Unknown"
    if isinstance(type_field, list) and type_field:
        type_field = type_field[0]
    type_str = str(type_field)
    return type_str.split(":")[-1] if ":" in type_str else type_str


def get_all_labels(field: Any) -> Optional[str]:
    """
    Collapse a JSON-LD multilingual/multivalue field into one string.
    """
    if not field:
        return None

    if isinstance(field, str):
        return field

    if isinstance(field, dict):
        return field.get("@value", str(field))

    if isinstance(field, list):
        values: List[str] = []
        for item in field:
            if isinstance(item, dict) and "@value" in item:
                values.append(str(item["@value"]))
            elif isinstance(item, str):
                values.append(item)
        return " / ".join(v.strip() for v in values if str(v).strip()) or None

    return str(field)


def get_lang_value(field: Any, lang: str) -> Optional[str]:
    """
    Return the first value of a multilingual JSON-LD field for a target language.
    """
    if not field:
        return None

    if isinstance(field, dict):
        if field.get("@language") == lang and "@value" in field:
            return str(field["@value"])
        return None

    if isinstance(field, list):
        for item in field:
            if isinstance(item, dict) and item.get("@language") == lang and "@value" in item:
                return str(item["@value"])

    return None


def extract_literal_values(field: Any) -> List[str]:
    """
    Extract text literals from JSON-LD-ish structures.
    """
    if not field:
        return []

    if isinstance(field, str):
        val = field.strip()
        return [val] if val else []

    if isinstance(field, dict):
        if "@value" in field:
            val = str(field["@value"]).strip()
            return [val] if val else []
        return []

    if isinstance(field, list):
        out: List[str] = []
        for item in field:
            out.extend(extract_literal_values(item))
        return [v for v in out if v]

    val = str(field).strip()
    return [val] if val else []


def extract_id_targets(field: Any) -> List[str]:
    """
    Extract URI/@id targets from JSON-LD-ish structures.
    """
    if not field:
        return []

    if isinstance(field, str):
        return [field] if field.startswith("http") else []

    if isinstance(field, dict):
        target_id = field.get("@id")
        return [target_id] if isinstance(target_id, str) and target_id.startswith("http") else []

    if isinstance(field, list):
        out: List[str] = []
        for item in field:
            out.extend(extract_id_targets(item))
        return out

    return []


def clean_date_to_int(date_field: Any) -> Optional[int]:
    """
    Parse TimeSpan begin/end values such as '-0031' or '0324' into ints.
    """
    if not date_field:
        return None

    val = date_field.get("@value") if isinstance(date_field, dict) else str(date_field)
    val = val.strip()

    try:
        return int(val)
    except ValueError:
        return None


# ============================================================
# Created-date parsing for ProvidedCHO
# ============================================================

def normalize_year_token(token: str) -> Optional[int]:
    """
    Convert tokens like:
      '101'
      '0324'
      '-0031'
      '31 B.C.'
      '324 A.D.'
    to integer years.

    BCE/BC => negative integer
    CE/AD => positive integer
    """
    if not token:
        return None

    t = token.strip()
    t_upper = t.upper()

    # BCE / BC
    m = re.search(r"(\d{1,4})\s*(BC|B\.C\.|BCE|B\.C\.E\.)", t_upper)
    if m:
        return -int(m.group(1))

    # CE / AD
    m = re.search(r"(\d{1,4})\s*(AD|A\.D\.|CE|C\.E\.)", t_upper)
    if m:
        return int(m.group(1))

    # plain integer
    if re.fullmatch(r"-?\d{1,6}", t):
        return int(t)

    return None


def parse_created_years(created_field: Any) -> Tuple[Optional[int], Optional[int], bool, Optional[str]]:
    """
    Parse a CHO created field into:
      (created_start_year, created_end_year, created_is_approximate, created_text)

    Handles forms like:
      '101 AD - 200 AD'
      '31 B.C. - 324 A.D.'
      '-0031'
      '0324'
      '101'
    """
    raw_text = get_all_labels(created_field)
    if not raw_text:
        return None, None, False, None

    text = raw_text.strip()
    is_approximate = False

    lower = text.lower()
    approx_markers = ["circa", "ca.", "ca ", "approx", "approximately", "?"]
    if any(marker in lower for marker in approx_markers):
        is_approximate = True

    normalized = (
        text.replace("–", "-")
            .replace("—", "-")
            .replace(" έως ", " - ")
            .replace(" to ", " - ")
    )

    # Try simple range split first
    parts = [p.strip() for p in re.split(r"\s*-\s*", normalized) if p.strip()]
    if len(parts) == 2:
        start = normalize_year_token(parts[0])
        end = normalize_year_token(parts[1])
        if start is not None and end is not None:
            if start <= end:
                return start, end, is_approximate, raw_text
            return end, start, is_approximate, raw_text

    # Fallback: scan for year-like expressions
    candidates: List[int] = []

    for m in re.finditer(r"(\d{1,4})\s*(BC|B\.C\.|BCE|B\.C\.E\.)", text, flags=re.I):
        candidates.append(-int(m.group(1)))

    for m in re.finditer(r"(\d{1,4})\s*(AD|A\.D\.|CE|C\.E\.)", text, flags=re.I):
        candidates.append(int(m.group(1)))

    for m in re.finditer(r"(?<!\d)(-?\d{1,4})(?!\d)", text):
        val = int(m.group(1))
        if val not in candidates:
            candidates.append(val)

    if len(candidates) >= 2:
        start, end = min(candidates), max(candidates)
        return start, end, is_approximate, raw_text

    if len(candidates) == 1:
        year = candidates[0]
        return year, year, is_approximate, raw_text

    return None, None, is_approximate, raw_text


# ============================================================
# Neo4j schema
# ============================================================

def ensure_constraints(tx) -> None:
    tx.run("""
        CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
        FOR (n:Entity) REQUIRE n.id IS UNIQUE
    """)
    tx.run("""
        CREATE CONSTRAINT external_id_unique IF NOT EXISTS
        FOR (n:External) REQUIRE n.id IS UNIQUE
    """)
    tx.run("""
        CREATE CONSTRAINT material_name_unique IF NOT EXISTS
        FOR (m:Material) REQUIRE m.name IS UNIQUE
    """)

    tx.run("""
        CREATE INDEX entity_label_idx IF NOT EXISTS
        FOR (n:Entity) ON (n.label)
    """)
    tx.run("""
        CREATE INDEX cho_identifier_idx IF NOT EXISTS
        FOR (n:ProvidedCHO) ON (n.identifier)
    """)
    tx.run("""
        CREATE INDEX cho_created_start_idx IF NOT EXISTS
        FOR (n:ProvidedCHO) ON (n.created_start_year)
    """)
    tx.run("""
        CREATE INDEX cho_created_end_idx IF NOT EXISTS
        FOR (n:ProvidedCHO) ON (n.created_end_year)
    """)
    tx.run("""
        CREATE INDEX timespan_begin_idx IF NOT EXISTS
        FOR (n:TimeSpan) ON (n.begin_year)
    """)
    tx.run("""
        CREATE INDEX timespan_end_idx IF NOT EXISTS
        FOR (n:TimeSpan) ON (n.end_year)
    """)


def clear_database(tx) -> None:
    tx.run("MATCH (n) DETACH DELETE n")


# ============================================================
# Ingestion phases
# ============================================================

def create_or_update_nodes(tx, graph_data: List[Dict[str, Any]]) -> None:
    for item in graph_data:
        node_id = item.get("@id")
        if not node_id:
            continue

        node_type = _safe_label(_type_suffix(item.get("@type")))
        chosen_label = get_all_labels(
            item.get("prefLabel") or item.get("title") or item.get("altLabel")
        ) or str(node_id)

        query = """
        MERGE (n:Entity {id: $node_id})
        SET n.label = $label
        SET n:$($node_type)
        """
        params: Dict[str, Any] = {
            "node_id": node_id,
            "label": chosen_label,
            "node_type": node_type,
        }

        if node_type == "ProvidedCHO":
            created_start, created_end, created_is_approx, created_text = parse_created_years(item.get("created"))

            query = """
            MERGE (n:Entity {id: $node_id})
            SET n.label = $label,
                n.title = $title,
                n.title_en = $title_en,
                n.title_el = $title_el,
                n.description = $description,
                n.created_text = $created_text,
                n.created_start_year = $created_start_year,
                n.created_end_year = $created_end_year,
                n.created_is_approximate = $created_is_approximate,
                n.extent = $extent,
                n.medium = $medium,
                n.artifact_type = $artifact_type,
                n.source = $source,
                n.identifier = $identifier,
                n.edm_type = $edm_type,
                n.dcterms_spatial = $dcterms_spatial
            SET n:$($node_type)
            """
            params.update({
                "title": get_all_labels(item.get("title")),
                "title_en": get_lang_value(item.get("title"), "en"),
                "title_el": get_lang_value(item.get("title"), "el"),
                "description": get_all_labels(item.get("description")),
                "created_text": created_text,
                "created_start_year": created_start,
                "created_end_year": created_end,
                "created_is_approximate": created_is_approx,
                "extent": get_all_labels(item.get("extent")),
                "medium": get_all_labels(item.get("medium")),
                "artifact_type": get_all_labels(item.get("dc:type")),
                "source": get_all_labels(item.get("source")),
                "identifier": get_all_labels(item.get("identifier")),
                "edm_type": get_all_labels(item.get("edm:type")),
                "dcterms_spatial": get_all_labels(item.get("dcterms:spatial")),
            })

        elif node_type == "TimeSpan":
            query = """
            MERGE (n:Entity {id: $node_id})
            SET n.label = $label,
                n.begin_year = $begin_year,
                n.end_year = $end_year
            SET n:$($node_type)
            """
            params.update({
                "begin_year": clean_date_to_int(item.get("begin")),
                "end_year": clean_date_to_int(item.get("end")),
            })

        elif node_type == "Aggregation":
            query = """
            MERGE (n:Entity {id: $node_id})
            SET n.label = $label,
                n.data_provider = $data_provider,
                n.provider = $provider,
                n.rights = $rights,
                n.image_url = $image_url,
                n.is_shown_at = $is_shown_at,
                n.object_url = $object_url,
                n.has_view = $has_view
            SET n:$($node_type)
            """
            params.update({
                "data_provider": get_all_labels(item.get("dataProvider")),
                "provider": get_all_labels(item.get("provider")),
                "rights": get_all_labels(item.get("rights")),
                "image_url": get_all_labels(item.get("isShownBy")),
                "is_shown_at": get_all_labels(item.get("isShownAt")),
                "object_url": get_all_labels(item.get("object")),
                "has_view": extract_id_targets(item.get("hasView")),
            })

        elif node_type == "Place":
            query = """
            MERGE (n:Entity {id: $node_id})
            SET n.label = $label,
                n.lat = $lat,
                n.long = $long
            SET n:$($node_type)
            """
            params.update({
                "lat": item.get("lat"),
                "long": item.get("long"),
            })

        tx.run(query, **params)

def create_entity_relationships(tx, graph_data: List[Dict[str, Any]]) -> None:
    """
    Second pass: create source-declared entity-to-entity relationships.
    """
    entity_relationship_fields: Dict[str, str] = {
        "subject": "HAS_SUBJECT",
        "type": "HAS_TYPE",
        "spatial": "LOCATED_IN",
        "broader": "BROADER_THAN",
        "narrower": "NARROWER_THAN",
        "broadMatch": "BROAD_MATCH",
        "aggregatedCHO": "AGGREGATES",
        "isPartOf": "IS_PART_OF",
        "temporal": "HAS_TEMPORAL_REFERENCE",
    }

    for item in graph_data:
        source_id = item.get("@id")
        if not source_id:
            continue

        for field, rel_name in entity_relationship_fields.items():
            targets = extract_id_targets(item.get(field))
            if not targets:
                continue

            for target_id in targets:
                tx.run(f"""
                    MATCH (source:Entity {{id: $source_id}})
                    MERGE (target:Entity {{id: $target_id}})
                    MERGE (source)-[:{rel_name}]->(target)
                """, source_id=source_id, target_id=target_id)


def create_external_relationships(tx, graph_data: List[Dict[str, Any]]) -> None:
    """
    Third pass: create entity-to-external URI relationships.
    """
    external_relationship_fields: Dict[str, str] = {
        "exactMatch": "EXACT_MATCH",
        "sameAs": "SAME_AS",
    }

    for item in graph_data:
        source_id = item.get("@id")
        if not source_id:
            continue

        for field, rel_name in external_relationship_fields.items():
            targets = extract_id_targets(item.get(field))
            if not targets:
                continue

            for target_id in targets:
                tx.run(f"""
                    MATCH (source:Entity {{id: $source_id}})
                    MERGE (ext:External {{id: $target_id}})
                    ON CREATE SET ext.label = $target_id
                    MERGE (source)-[:{rel_name}]->(ext)
                """, source_id=source_id, target_id=target_id)


def create_material_nodes(tx, graph_data: List[Dict[str, Any]]) -> None:
    """
    Fourth pass: normalize medium literals into Material nodes.
    """
    for item in graph_data:
        if _type_suffix(item.get("@type")) != "ProvidedCHO":
            continue

        source_id = item.get("@id")
        if not source_id:
            continue

        for material_name in extract_literal_values(item.get("medium")):
            if material_name.startswith("http"):
                continue

            tx.run("""
                MATCH (source:Entity:ProvidedCHO {id: $source_id})
                MERGE (m:Material {name: $name})
                MERGE (source)-[:MADE_OF]->(m)
            """, source_id=source_id, name=material_name)


def ingest_graph_chunk(tx, graph_data: List[Dict[str, Any]]) -> None:
    """
    Full ingestion for one JSONL object's @graph payload.
    """
    create_or_update_nodes(tx, graph_data)
    create_entity_relationships(tx, graph_data)
    create_external_relationships(tx, graph_data)
    create_material_nodes(tx, graph_data)


# ============================================================
# Post-processing
# ============================================================

def build_from_period_relationships(tx) -> None:
    """
    Create retrieval-friendly FROM_PERIOD edges based on date overlap.

    Semantics:
      (cho)-[:FROM_PERIOD]->(period)
    means that the artifact's normalized created date range overlaps
    the TimeSpan's [begin_year, end_year].
    """
    tx.run("""
        MATCH (cho:Entity:ProvidedCHO)
        WHERE cho.created_start_year IS NOT NULL
          AND cho.created_end_year IS NOT NULL
        MATCH (period:Entity:TimeSpan)
        WHERE period.begin_year IS NOT NULL
          AND period.end_year IS NOT NULL
          AND cho.created_end_year >= period.begin_year
          AND cho.created_start_year <= period.end_year
        MERGE (cho)-[:FROM_PERIOD]->(period)
    """)


def remove_from_period_relationships(tx) -> None:
    """
    Remove inferred FROM_PERIOD relationships before rebuilding them.
    Safe to call repeatedly.
    """
    tx.run("MATCH ()-[r:FROM_PERIOD]->() DELETE r")


# ============================================================
# Main
# ============================================================

def main() -> None:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    file_path = os.getenv("JSONL_PATH")
    start_line = int(os.getenv("START_LINE", "0"))
    reset_db = os.getenv("RESET_DB", "false").lower() == "true"

    if not uri or not user or not password or not file_path:
        raise RuntimeError("Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, and JSONL_PATH.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    print("Calculating total items to ingest...")
    with open(file_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    auth = (user, password)

    with GraphDatabase.driver(uri, auth=auth) as driver:
        with driver.session() as session:
            if reset_db:
                print("RESET_DB=true -> clearing database...")
                session.execute_write(clear_database)

            print("Ensuring constraints and indexes...")
            session.execute_write(ensure_constraints)

            print(f"Starting ingestion from line {start_line}...")
            with open(file_path, "r", encoding="utf-8") as f:
                skipped_file = itertools.islice(f, start_line, None)

                for line in tqdm(
                    skipped_file,
                    initial=start_line,
                    total=total_lines,
                    desc="Ingesting JSONL into Neo4j",
                    unit="item",
                ):
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)
                    graph_data = data.get("@graph", [])
                    session.execute_write(ingest_graph_chunk, graph_data)

            print("Rebuilding inferred FROM_PERIOD relationships...")
            session.execute_write(remove_from_period_relationships)
            session.execute_write(build_from_period_relationships)

    print("Done. Graph ingested and FROM_PERIOD rebuilt.")


if __name__ == "__main__":
    main()
