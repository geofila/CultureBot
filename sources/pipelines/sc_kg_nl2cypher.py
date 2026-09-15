"""
SearchCulture.gr NL → Cypher helper utilities.
Generated from SearchCulture advanced-search HTML and designed for Neo4j KG querying.

Core idea:
- Keep the full SearchCulture filter/taxonomy info on disk.
- Inject only the relevant filter candidates into the LLM prompt.
- For high-level places, match the whole taxonomy branch through path/prefix candidates
  and fallback textual fields, rather than relying on one leaf place.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except Exception:  # lets the module import even before dependency setup
    OpenAI = None

try:
    from neo4j import GraphDatabase
except Exception:
    GraphDatabase = None


# In the deployed image this module lives in /app/pipelines/kg_jsons/ (see
# Dockerfile.pipelines), so ROOT is that kg_jsons dir: the taxonomy/filter JSONs are
# mounted next to it, while the text-chunk files are mounted one level up in
# /app/pipelines/. Every path can be overridden with an env var for other layouts.
ROOT = Path(__file__).resolve().parent
PIPELINES_DIR = ROOT.parent  # /app/pipelines in the container (where *_chunks.* are mounted)

PLACES_JSON = Path(os.getenv("PLACES_JSON") or (ROOT / "searchculture_places_taxonomy.json"))
FILTERS_JSON = Path(os.getenv("FILTERS_JSON") or (ROOT / "searchculture_filters_from_page.json"))

# All SearchCulture advanced-search filters visible in the uploaded page.
# These are mapped to Neo4j patterns/properties used by the KG.
FILTER_SPECS: Dict[str, Dict[str, Any]] = {
    "keywords": {
        "label": "Πληκτρολογήστε λέξη ή φράση / keyword",
        "cypher": "item.label, item.title, item.title_el, item.title_en, item.description",
        "notes": "Use for free text when no controlled filter is clearly requested.",
    },
    "title": {
        "label": "Τίτλος / title",
        "cypher": "item.title, item.title_el, item.title_en, item.label",
    },
    "type": {
        "label": "Τύπος τεκμηρίου / item type",
        "cypher": "(item)-[:HAS_TYPE]->(type:Entity:Concept); type.label; also item.artifact_type",
    },
    "subject": {
        "label": "Θέμα / subject",
        "cypher": "(item)-[:HAS_SUBJECT]->(subject:Entity:Concept); subject.label",
    },
    "material": {
        "label": "Υλικό / material",
        "cypher": "(item)-[:MADE_OF]->(m:Material); m.name (single-language node, e.g. 'Μάρμαρο' and 'Marble' are two separate Material nodes — OR both); fallback item.medium",
    },
    "person": {
        "label": "Πρόσωπο / person, as creator or reference",
        "cypher": "Agent nodes are NOT connected to ProvidedCHO in the graph (only to External via SAME_AS). Always use item.source/item.description/item.label/item.title free-text fallback for persons.",
    },
    "profession_or_occupation": {
        "label": "Ιδιότητα προσώπου / profession or occupation",
        "cypher": "No dedicated relationship; use free-text fallback over label/title/description.",
    },
    "historical_period": {
        "label": "Ιστορική περίοδος / historical period",
        "cypher": "(item)-[:HAS_TEMPORAL_REFERENCE|FROM_PERIOD]->(t:Entity:TimeSpan); t.label, t.begin_year, t.end_year",
        "rules": "Both HAS_TEMPORAL_REFERENCE and FROM_PERIOD exist in the live graph between ProvidedCHO and TimeSpan; match through both relationship types (e.g. -[:HAS_TEMPORAL_REFERENCE|FROM_PERIOD]->) to avoid missing rows.",
    },
    "chronology": {
        "label": "Χρονολόγηση / year span",
        "cypher": "item.created_start_year, item.created_end_year, item.created_text",
        "rules": "For overlap [start,end]: item.created_start_year <= end AND item.created_end_year >= start. For strict: start <= item.created_start_year AND item.created_end_year <= end.",
    },
    "place": {
        "label": "Τόπος / place",
        "cypher": "(item)-[:LOCATED_IN]->(p:Entity:Place); p.label only (single combined field, often 'Transliteration / Greek' or similar — no pref_label_el/en/alt_labels exist); fallback item.dcterms_spatial (plain STRING, not a list — use CONTAINS directly, not ANY/IN), item.description, item.label, item.title",
        "rules": "For broad/high-level place requests, match the place itself plus descendants using injected taxonomy candidates and path labels, all against p.label.",
    },
    "provider_or_collection": {
        "label": "Φορέας / συλλογή",
        "cypher": "(agg:Entity:Aggregation)-[:AGGREGATES]->(item); agg.data_provider, agg.provider",
    },
    "rights": {
        "label": "Άδεια χρήσης αρχείου / rights",
        "cypher": "(agg:Entity:Aggregation)-[:AGGREGATES]->(item); agg.rights",
    },
    "image_resolution": {
        "label": "Ανάλυση (για εικόνες, video ή PDF)",
        "cypher": "Usually not in KG unless stored as media metadata; prefer Aggregation/media fields if available.",
    },
    "file_format": {
        "label": "Τύπος αρχείου / format",
        "cypher": "Use media/file metadata if available; otherwise Aggregation object/image URL fields.",
    },
    "edm_type": {
        "label": "EDM type: IMAGE/TEXT/VIDEO/SOUND/3D",
        "cypher": "item.edm_type",
    },
    "has_image_or_landing_page": {
        "label": "image URL / landing page",
        "cypher": "(agg:Entity:Aggregation)-[:AGGREGATES]->(item); agg.image_url, agg.is_shown_at, agg.object_url, agg.has_view",
    },
}

SCHEMA_SUMMARY = """
Neo4j schema summary (verified directly against the live database — exact property names matter, unknown properties silently return null instead of erroring):
Node labels:
- Entity:ProvidedCHO: cultural heritage object/item (81,803 nodes).
- Entity:Place: controlled place (1,249 nodes).
- Entity:TimeSpan: historical period/time span (44 nodes).
- Entity:Concept: type/subject concept (167 nodes).
- Entity:Aggregation: provider/media/rights/web representation (81,803 nodes).
- Material: material/medium concept, NOT under Entity (628 nodes).
- Agent: person/creator authority (62 nodes) — NOT connected to ProvidedCHO; ignore for item queries.

ProvidedCHO properties (all single-valued strings/ints/bools — there are NO "_values" plural/array properties):
- id, label, title, title_el, title_en, description
- created_text, created_start_year (int), created_end_year (int), created_is_approximate (bool)
- extent, medium, artifact_type
- source, identifier, edm_type, dcterms_spatial (plain STRING, not a list)

Place properties: id, label, lat, long.
Place has ONLY `label` — no pref_label_el, pref_label_en, or alt_labels. `label` is a single string that
often combines transliteration/Greek, e.g. "Fílippoi / Φίλιπποι". Search both Greek and English query terms
with CONTAINS against this one `label` field.

TimeSpan properties: id, label, begin_year (int), end_year (int).
Same caveat: TimeSpan has ONLY `label` (combined bilingual string, e.g. "Ρωμαϊκή περίοδος / Roman Period") —
no pref_label_el/en/alt_labels. Use begin_year/end_year for numeric period ranges.

Concept properties: id, label. Same caveat — no pref_label_el/en split fields.

Material properties: name (single-language string, e.g. "Μάρμαρο" and "Marble" are two SEPARATE nodes for
the same material — match both language variants with OR when relevant).

Aggregation properties: id, label, data_provider, provider, rights, image_url, is_shown_at, object_url, has_view.

Relationships:
- (:Entity:ProvidedCHO)-[:LOCATED_IN]->(:Entity:Place)
- (:Entity:ProvidedCHO)-[:HAS_TEMPORAL_REFERENCE]->(:Entity:TimeSpan)  -- canonical/cleaned period, ~1.5 per item
- (:Entity:ProvidedCHO)-[:FROM_PERIOD]->(:Entity:TimeSpan)  -- also present, more numerous (~4 per item); match BOTH with -[:HAS_TEMPORAL_REFERENCE|FROM_PERIOD]->
- (:Entity:ProvidedCHO)-[:HAS_TYPE]->(:Entity:Concept)
- (:Entity:ProvidedCHO)-[:HAS_SUBJECT]->(:Entity:Concept)
- (:Entity:ProvidedCHO)-[:MADE_OF]->(:Material)
- (:Entity:Aggregation)-[:AGGREGATES]->(:Entity:ProvidedCHO)
- (:Entity:Place)-[:IS_PART_OF]->(:Entity) — sparse (1,273 edges over 1,249 places); do not rely on this for
  place hierarchy, use the injected taxonomy candidates below instead.
""".strip()

FEW_SHOTS = [
    {
        "user": "φέρε τεκμήρια από την αρχαία Μεσσήνη",
        "cypher": """MATCH (item:Entity:ProvidedCHO)
WHERE EXISTS {
  MATCH (item)-[:LOCATED_IN]->(p:Entity:Place)
  WHERE toLower(p.label) CONTAINS toLower('Αρχαία Μεσσήνη')
     OR toLower(p.label) CONTAINS toLower('Μεσσήνη')
     OR toLower(p.label) CONTAINS toLower('Ancient Messene')
     OR toLower(p.label) CONTAINS toLower('Messene')
}
   OR toLower(coalesce(item.dcterms_spatial,'')) CONTAINS toLower('Αρχαία Μεσσήνη')
   OR toLower(coalesce(item.dcterms_spatial,'')) CONTAINS toLower('Μεσσήνη')
   OR toLower(coalesce(item.description,'')) CONTAINS toLower('Αρχαία Μεσσήνη')
RETURN DISTINCT item.id AS item_id, item.label AS item_label, item.created_text AS created_text
LIMIT 100""",
    },
    {
        "user": "φέρε όλα τα μνημεία από τους τόπους της Θεσσαλονίκης",
        "cypher": """MATCH (item:Entity:ProvidedCHO)
WHERE EXISTS {
  MATCH (item)-[:LOCATED_IN]->(p:Entity:Place)
  WHERE toLower(p.label) CONTAINS toLower('Θεσσαλονίκη')
     OR toLower(p.label) CONTAINS toLower('Θεσσαλονίκης')
     OR toLower(p.label) CONTAINS toLower('Thessaloniki')
}
   OR toLower(coalesce(item.dcterms_spatial,'')) CONTAINS toLower('Θεσσαλονίκη')
   OR toLower(coalesce(item.dcterms_spatial,'')) CONTAINS toLower('Θεσσαλονίκης')
   OR toLower(coalesce(item.dcterms_spatial,'')) CONTAINS toLower('Thessaloniki')
RETURN DISTINCT item.id AS item_id, item.label AS item_label, item.created_text AS created_text
LIMIT 100""",
    },
    {
        "user": "ποιες είναι οι πιο συνηθισμένες μορφές στα νομίσματα της Κάτω Ιταλίας;",
        "cypher": """MATCH (item:Entity:ProvidedCHO)
WHERE (
    EXISTS {
      MATCH (item)-[:LOCATED_IN]->(p:Entity:Place)
      WHERE toLower(p.label) CONTAINS toLower('Κάτω Ιταλία')
         OR toLower(p.label) CONTAINS toLower('Νότια Ιταλία')
         OR toLower(p.label) CONTAINS toLower('South Italy')
         OR toLower(p.label) CONTAINS toLower('Southern Italy')
         OR toLower(p.label) CONTAINS toLower('Magna Graecia')
         OR toLower(p.label) CONTAINS toLower('Μεγάλη Ελλάδα')
    }
 OR toLower(coalesce(item.dcterms_spatial,'')) CONTAINS toLower('Κάτω Ιταλία')
 OR toLower(coalesce(item.dcterms_spatial,'')) CONTAINS toLower('Νότια Ιταλία')
 OR toLower(coalesce(item.dcterms_spatial,'')) CONTAINS toLower('Magna Graecia')
 OR toLower(coalesce(item.dcterms_spatial,'')) CONTAINS toLower('Μεγάλη Ελλάδα')
)
AND (
    toLower(coalesce(item.artifact_type,'')) CONTAINS toLower('νόμισμα')
 OR toLower(coalesce(item.label,'')) CONTAINS toLower('νόμισμα')
 OR toLower(coalesce(item.description,'')) CONTAINS toLower('νόμισμα')
 OR toLower(coalesce(item.description,'')) CONTAINS toLower('coin')
)
RETURN DISTINCT item.id AS item_id, item.label AS item_label, item.description AS description, item.created_text AS created_text
LIMIT 200""",
    },
    {
        "user": "δείξε βυζαντινές εικόνες με διαθέσιμο image url",
        "cypher": """MATCH (item:Entity:ProvidedCHO)-[:HAS_TEMPORAL_REFERENCE|FROM_PERIOD]->(t:Entity:TimeSpan)
MATCH (agg:Entity:Aggregation)-[:AGGREGATES]->(item)
WHERE (toLower(coalesce(t.label,'')) CONTAINS toLower('Βυζαντινή') OR toLower(coalesce(t.label,'')) CONTAINS toLower('Byzantine'))
  AND agg.image_url IS NOT NULL
RETURN DISTINCT item.id AS item_id, item.label AS item_label, t.label AS period, agg.image_url AS image_url, agg.is_shown_at AS landing_page
LIMIT 100""",
    },
    {
        "user": "μαρμάρινα γλυπτά από Αθήνα 100 π.Χ. έως 200 μ.Χ.",
        "cypher": """MATCH (item:Entity:ProvidedCHO)
WHERE EXISTS {
        MATCH (item)-[:LOCATED_IN]->(p:Entity:Place)
        WHERE toLower(p.label) CONTAINS toLower('Αθήνα') OR toLower(p.label) CONTAINS toLower('Athens')
      }
  AND (
        EXISTS {
          MATCH (item)-[:HAS_TYPE]->(type:Entity:Concept)
          WHERE toLower(type.label) CONTAINS toLower('γλυπτό') OR toLower(type.label) CONTAINS toLower('sculpture')
        }
     OR toLower(coalesce(item.artifact_type,'')) CONTAINS toLower('γλυπτό')
      )
  AND (
        EXISTS {
          MATCH (item)-[:MADE_OF]->(mat:Material)
          WHERE toLower(mat.name) CONTAINS toLower('μάρμαρο') OR toLower(mat.name) CONTAINS toLower('marble')
        }
     OR toLower(coalesce(item.medium,'')) CONTAINS toLower('μάρμαρο')
      )
  AND item.created_start_year IS NOT NULL AND item.created_end_year IS NOT NULL
  AND item.created_start_year <= 200 AND item.created_end_year >= -100
RETURN DISTINCT item.id AS item_id, item.label AS item_label, item.medium AS medium, item.created_text AS created_text
LIMIT 100""",
    },
]


def strip_accents(text: str) -> str:
    text = unicodedata.normalize('NFD', text or '')
    return ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')


def norm(text: str) -> str:
    text = strip_accents(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compact_terms(text: str) -> List[str]:
    """Terms useful for fuzzy place matching; keeps specific Greek/English stems > 4 chars."""
    n = norm(text)
    stop = {
        'φερε','ολα','ολοι','ολες','δειξε','θελω','τεκμηρια','μνημεια','τοπους','τοπος',
        'αρχαια','αρχαιος','αρχαιο','ancient','monuments','items','records','from','places',
        'νομισματα','νομισμα','coins','coin','συνηθισμενες','μορφες','ποιες','ειναι'
    }
    terms = [t for t in n.split() if len(t) >= 5 and t not in stop]
    # crude Greek genitive handling: θεσσαλονικης -> θεσσαλονικ
    stems = set(terms)
    for t in terms:
        for suf in ('ης','ας','ου','ων','ος'):
            nsuf = norm(suf)
            if t.endswith(nsuf) and len(t) > len(nsuf)+5:
                stems.add(t[:-len(nsuf)])
    return sorted(stems, key=len, reverse=True)


def load_places(path: Path | str = PLACES_JSON) -> List[Dict[str, Any]]:
    # Missing or malformed taxonomy is not fatal: the Cypher prompt then simply carries no
    # place candidates. The pipelines normally pass places= explicitly (discovered from the
    # dataset folder by dataset_loader), so this default path is for standalone/notebook use.
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        print(f"[sc_kg_nl2cypher] no place taxonomy at {path} ({exc}); continuing without one")
        return []


def load_filters(path: Path | str = FILTERS_JSON) -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# A node whose own name is explicitly named in the query is treated as a hierarchy
# "anchor": every descendant of that node (by taxonomy path containment) is pulled in
# unconditionally, not subject to the fuzzy score cap below. This is what keeps broad
# requests like "τόποι της Θεσσαλονίκης" or "Νομός Θεσσαλονίκης" matching the *whole*
# branch instead of whatever 24 leaves happened to score highest.
MIN_ANCHOR_LABEL_LEN = 4
MAX_ANCHOR_DESCENDANTS = 150  # guards against mega-anchors like "Ελλάδα"/"Ευρώπη" (1000+ descendants)


def find_place_anchors(query: str, places: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Nodes whose own label/alt_label is named verbatim (as a phrase) in the query."""
    qn = norm(query)
    anchors = []
    for p in places:
        fields = [p.get('label_el', ''), p.get('label_en', '')] + list(p.get('alt_labels') or [])
        for f in fields:
            fn = norm(f)
            if fn and len(fn) >= MIN_ANCHOR_LABEL_LEN and fn in qn:
                anchors.append(p)
                break
    return anchors


def find_place_descendants(anchor: Dict[str, Any], places: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """All taxonomy nodes whose path passes through `anchor` (by label_el or label_en)."""
    anchor_keys = {anchor.get('label_el'), anchor.get('label_en')} - {None, ''}
    anchor_depth = int(anchor.get('depth') or 0)
    out = []
    for p in places:
        if p is anchor:
            continue
        path = p.get('path') or []
        if int(p.get('depth') or 0) > anchor_depth and any(k in path for k in anchor_keys):
            out.append(p)
    return out


def find_place_candidates(query: str, places: Optional[List[Dict[str, Any]]] = None, max_candidates: int = 24) -> List[Dict[str, Any]]:
    """
    Return place candidates for the query.

    Two complementary mechanisms:
    1. Hierarchy anchors: if the query names a taxonomy node directly (e.g. "Νομός
       Θεσσαλονίκης"), that node plus ALL of its descendants are included in full —
       this is what "maintains the hierarchy below" for place filters. Skipped for
       mega-anchors (e.g. "Ελλάδα") whose branch is effectively the whole dataset.
    2. Fuzzy fallback: scored term/path matching for partial or loose mentions,
       capped at max_candidates, used to fill in anything anchors didn't cover.
    """
    if places is None:
        places = load_places()

    out: List[Dict[str, Any]] = []
    seen = set()

    def add(p: Dict[str, Any]) -> None:
        key = (p.get('label_el'), p.get('label_en'))
        if key not in seen:
            seen.add(key)
            out.append(p)

    anchors = find_place_anchors(query, places)
    for anchor in anchors:
        descendants = find_place_descendants(anchor, places)
        if len(descendants) > MAX_ANCHOR_DESCENDANTS:
            # Anchor too broad (e.g. "Ελλάδα"/"Ευρώπη") to usefully enumerate; fall through
            # to fuzzy matching instead of dumping ~1000 candidates into the prompt.
            continue
        add(anchor)
        for d in descendants:
            add(d)

    qn = norm(query)
    terms = compact_terms(query)
    scored = []
    for p in places:
        key = (p.get('label_el'), p.get('label_en'))
        if key in seen:
            continue
        fields = [p.get('label_el', ''), p.get('label_en', '')] + list(p.get('alt_labels') or []) + list(p.get('path') or [])
        field_norms = [norm(x) for x in fields if x]
        path_norm = norm(p.get('path_text', ''))
        score = 0
        # exact phrase / label match
        for fn in field_norms:
            if fn and fn in qn:
                score += 100 + len(fn) / 10
            if qn and qn in fn and len(qn) > 4:
                score += 80
        # stem/term match over path catches "Θεσσαλονίκης" etc.
        # Use token-prefix matching to avoid false positives like Ιταλίας -> Επιτάλιο.
        tokenized_fields = []
        for fn in field_norms:
            tokenized_fields.extend(fn.split())
        path_tokens = path_norm.split()
        for t in terms:
            if len(t) >= 5 and any(tok.startswith(t) or t == tok for tok in tokenized_fields):
                score += 20 + len(t) / 10
            if any(tok.startswith(t) or t == tok for tok in path_tokens):
                score += 10
        if score:
            # Prefer broader nodes if the query asks broad area, but keep strong exact leafs too.
            depth = int(p.get('depth') or 99)
            count = p.get('count') or 0
            scored.append((score + min(count, 5000) / 10000 - depth / 100, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    fuzzy_added = 0
    for _, p in scored:
        if fuzzy_added >= max_candidates:
            break
        add(p)
        fuzzy_added += 1
    return out


def build_place_prompt_block(query: str, places: Optional[List[Dict[str, Any]]] = None, max_candidates: int = 24) -> str:
    candidates = find_place_candidates(query, places, max_candidates=max_candidates)
    if not candidates:
        return ""
    lines = [
        "Relevant controlled place candidates from SearchCulture taxonomy.",
        "Use them as Greek/English place aliases.",
        "If candidates below share a common ancestor branch (e.g. a region/prefecture and its towns/sites),",
        "this list is already the COMPLETE branch — you MUST add an OR-condition for every single one of them",
        "(matching p.label with CONTAINS for both the Greek and English form — Place has ONLY a `label`",
        "property, no pref_label_el/en/alt_labels), not just the top-level name.",
        "Do not invent place names that are not in this list and not the user's own literal wording.",
        "Format: Greek | English | taxonomy path | count",
    ]
    for p in candidates:
        lines.append(f"- {p.get('label_el','')} | {p.get('label_en','')} | {p.get('path_text','')} | {p.get('count','')}")
    return "\n".join(lines)


def build_filter_prompt_block(max_values_per_filter: int = 16) -> str:
    """Compact description of all filters. Does not dump all facet values."""
    lines = ["SearchCulture advanced-search filters available to map from natural language:"]
    for key, spec in FILTER_SPECS.items():
        lines.append(f"- {key}: {spec.get('label')} → {spec.get('cypher')}")
        if spec.get('rules'):
            lines.append(f"  Rule: {spec['rules']}")
    return "\n".join(lines)


def build_few_shot_block() -> str:
    parts=[]
    for i, ex in enumerate(FEW_SHOTS, 1):
        parts.append(f"Example {i}\nUser: {ex['user']}\nCypher:\n{ex['cypher']}")
    return "\n\n".join(parts)


def build_cypher_prompt(user_query: str, limit: int = 100, places: Optional[List[Dict[str, Any]]] = None) -> str:
    # places: pass a preloaded list (e.g. from a long-lived service) to skip re-reading
    # PLACES_JSON from disk on every call; defaults to a fresh load_places() if omitted.
    place_block = build_place_prompt_block(user_query, places=places)
    return f"""
You are an expert Cypher generator for a Neo4j cultural heritage Knowledge Graph.
Generate exactly one READ-ONLY Cypher query for the user's Greek or English question.

{SCHEMA_SUMMARY}

{build_filter_prompt_block()}

Place matching rules — critical:
1. Place nodes have ONLY a `label` property (no pref_label_el/en, no alt_labels — those do not exist on the
   live graph and silently match nothing if used). p.label is one string that mixes Greek/English, e.g.
   "Fílippoi / Φίλιπποι". Always CONTAINS-match p.label against both the Greek and English forms of the name.
2. Also use fallback fields: item.dcterms_spatial (plain string, use CONTAINS — it is NOT a list), item.description, item.label, item.title.
3. For high-level/broad places like "Θεσσαλονίκη", "Νομός Θεσσαλονίκης", "Πελοπόννησος", "Κάτω Ιταλία", include all relevant branch/path aliases from the candidate list below. Do not restrict to one leaf.
4. For Greek without accents or genitive forms, search both variants where possible: Θεσσαλονίκη/Θεσσαλονίκης, Αθήνα/Αθηνα, Μεσσήνη/Μεσσηνη.
5. If the user asks for "όλα", "φέρε", "δείξε", return up to LIMIT {limit} unless another limit is requested.
6. If the user asks for "πιο συνηθισμένες", "συχνότερες", "πόσα", "ανά κατηγορία", prefer returning enough rows/properties for downstream aggregation, or use Cypher aggregation when the grouping field is explicit.
7. TimeSpan and Concept nodes also have ONLY `label` (no pref_label_el/en split) — same CONTAINS-both-languages approach.
8. Historical periods: match through BOTH relationship types, e.g. -[:HAS_TEMPORAL_REFERENCE|FROM_PERIOD]->(t:Entity:TimeSpan).
9. Materials: prefer (item)-[:MADE_OF]->(m:Material) with m.name CONTAINS (try both the Greek and English material name as separate Material nodes), falling back to item.medium.

{place_block}

Few-shot examples:
{build_few_shot_block()}

Safety and syntax rules:
- Output ONLY Cypher. No markdown, no explanation, no backticks.
- Query must start with MATCH or OPTIONAL MATCH.
- Do not use CREATE, MERGE, DELETE, SET, REMOVE, DROP, CALL dbms, LOAD CSV, APOC write procedures.
- Always RETURN item.id AS item_id and item.label AS item_label when returning items.
- Use DISTINCT to avoid duplicates.
- Always include LIMIT {limit}, unless doing a pure COUNT/GROUP aggregation.
- CRITICAL — never write `OPTIONAL MATCH (item)-[:REL]->(x) WHERE <condition on x>` as a filter. In Cypher this
  does NOT exclude rows: when no x satisfies the condition, the OPTIONAL MATCH still returns the row with x = NULL,
  so the filter is silently ignored and the query returns unrelated items. Instead, express "item is connected to
  something matching a condition" as a boolean using `EXISTS {{ MATCH (item)-[:REL]->(x) WHERE <condition> }}`,
  and combine it with OR/AND like any other boolean expression (see the few-shot examples below — every one of
  them uses this EXISTS {{ ... }} pattern for relationship-based filters, optionally OR'd with item-level
  fallback text fields). Only use OPTIONAL MATCH for relationships whose matched node/properties you want to
  RETURN (e.g. image_url, landing_page) and never put a row-filtering WHERE condition on its variables.

User question:
{user_query}
""".strip()


def strip_code_fences(text: str) -> str:
    text = (text or '').strip()
    text = re.sub(r'^```(?:cypher)?\s*', '', text, flags=re.I)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


def validate_cypher_text(cypher: str) -> Tuple[bool, str]:
    c = strip_code_fences(cypher)
    low = c.lower()
    if not (low.startswith('match') or low.startswith('optional match')):
        return False, 'Cypher must start with MATCH or OPTIONAL MATCH.'
    # Use \b word-boundary matching, not space-padding: generated Cypher is multi-line,
    # so a keyword right after a newline (e.g. "...\nRETURN ...") has no literal space
    # before it and a naive ' return ' substring check never matches.
    if not re.search(r'\breturn\b', low):
        return False, 'Cypher must contain RETURN.'
    if c.count('(') != c.count(')'):
        return False, 'Unbalanced parentheses.'
    forbidden = [r'\bcreate\b', r'\bmerge\b', r'\bdelete\b', r'\bset\b', r'\bremove\b', r'\bdrop\b', r'\bload\s+csv\b', r'\bcall\s+dbms\b', r'apoc\.create', r'apoc\.merge', r'apoc\.periodic']
    for pattern in forbidden:
        if re.search(pattern, low):
            return False, f'Forbidden write/admin operation detected: {pattern.strip()}'
    return True, c


def make_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    if OpenAI is None:
        raise ImportError('openai package is not installed. Run: pip install openai')
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('Set OPENAI_API_KEY environment variable or pass api_key=...')
    return OpenAI(api_key=api_key, base_url=base_url or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1')


def generate_cypher_with_openai(user_query: str, model: str = 'gpt-5.5', limit: int = 100, client=None) -> str:
    if client is None:
        client = make_openai_client()
    prompt = build_cypher_prompt(user_query, limit=limit)
    resp = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {'role': 'system', 'content': 'You generate precise, read-only Neo4j Cypher for cultural heritage search.'},
            {'role': 'user', 'content': prompt},
        ],
        # temperature=0.0,
    )
    cypher = strip_code_fences(resp.choices[0].message.content)
    ok, msg = validate_cypher_text(cypher)
    if not ok:
        raise ValueError(msg + '\n' + cypher)
    return msg


def connect_neo4j(uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
    if GraphDatabase is None:
        raise ImportError('neo4j package is not installed. Run: pip install neo4j')
    uri = uri or os.getenv('NEO4J_URI')
    user = user or os.getenv('NEO4J_USER', 'neo4j')
    password = password if password is not None else os.getenv('NEO4J_PASSWORD', '')
    if not uri:
        raise ValueError('Set NEO4J_URI environment variable or pass uri=...')
    return GraphDatabase.driver(uri, auth=(user, password))


def explain_cypher(driver, cypher: str) -> None:
    ok, c = validate_cypher_text(cypher)
    if not ok:
        raise ValueError(c)
    with driver.session() as session:
        session.run('EXPLAIN ' + c).consume()


def run_cypher(driver, cypher: str, max_rows: int = 200) -> List[Dict[str, Any]]:
    ok, c = validate_cypher_text(cypher)
    if not ok:
        raise ValueError(c)
    with driver.session() as session:
        result = session.run(c)
        rows = []
        for i, r in enumerate(result):
            rows.append(dict(r))
            if i + 1 >= max_rows:
                break
        return rows


def estimate_tokens(text: str, model_hint: str = 'gpt-5.5') -> int:
    """Approximate token count without tokenizer dependency. Greek often costs more than English; use this as planning estimate only."""
    if not text:
        return 0
    # Count words, punctuation and Greek chars. Conservative for mixed Greek/English prompt text.
    words = re.findall(r'\w+|[^\w\s]', text, flags=re.UNICODE)
    greek_chars = len(re.findall(r'[\u0370-\u03FF]', text))
    ascii_chars = len(text) - greek_chars
    return int(max(len(words) * 1.35, greek_chars / 2.2 + ascii_chars / 4.0))


# \u2500\u2500 Answer generation (mirrors KGSearchCultureBot.py's _ask_kg pipeline) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# Cypher rows only carry item_id/item_label \u2014 the actual descriptive text lives in
# puretext_chunks.jsonl, keyed by the same artifact id. Looking that text up and
# feeding it to the LLM is what turns raw graph rows into the same kind of grounded
# narrative answer the old bot produced, so the two pipelines can be compared head to head.

# puretext_chunks.jsonl is mounted at /app/pipelines/ (PIPELINES_DIR), one level up from
# this module's kg_jsons/ dir — NOT next to the module. Overridable via the JSONL_PATH env var.
JSONL_PATH = Path(os.getenv("JSONL_PATH") or (PIPELINES_DIR / "puretext_chunks.jsonl"))


def load_jsonl_records(path: Path | str = JSONL_PATH) -> Dict[str, Dict[str, Any]]:
    """
    {id -> {id, text}} lookup, mirroring _load_jsonl_records in KGSearchCultureBot.py.
    Supports {"id"/"uri": ..., "text"/"content"/"chunk_text"/"chunk": ...} records.
    This file is large (100MB+); call once and reuse the returned dict.
    """
    path = Path(path)
    records: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return records
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            rec_id = str(obj.get('id') or obj.get('uri') or '').strip()
            rec_text = str(
                obj.get('text') or obj.get('content') or obj.get('chunk_text') or obj.get('chunk') or ''
            ).strip()
            if not rec_id or not rec_text:
                continue
            records[rec_id] = {'id': rec_id, 'text': rec_text}
    return records


def extract_item_ids(graph_rows: List[Dict[str, Any]]) -> List[str]:
    """Mirrors _extract_item_ids: dedupe item_id values from Cypher result rows, in order."""
    ids: List[str] = []
    for row in graph_rows:
        item_id = row.get('item_id')
        if item_id and item_id not in ids:
            ids.append(item_id)
    return ids


def lookup_records_by_ids(records_by_id: Dict[str, Dict[str, Any]], item_ids: List[str]) -> List[Dict[str, Any]]:
    """Mirrors _lookup_records_by_ids: dedupe + map item_ids to their JSONL text record."""
    seen: set = set()
    matched: List[Dict[str, Any]] = []
    for item_id in item_ids:
        if item_id in seen:
            continue
        seen.add(item_id)
        record = records_by_id.get(item_id)
        if record is not None:
            matched.append(record)
    return matched


def build_kg_context(
    matched_records: List[Dict[str, Any]],
    graph_rows: List[Dict[str, Any]],
    max_context_records: int = 10,
    max_context_chars: int = 30000,
) -> str:
    """Mirrors _build_kg_context: graph rows (for structured fields) + matched JSONL full text."""
    parts = ["## Graph results"]
    parts.append("\n".join(
        f"[Graph Row {i}] {json.dumps(row, ensure_ascii=False)}"
        for i, row in enumerate(graph_rows[:max_context_records], start=1)
    ))
    parts.append("\n## Matched records")
    total_chars = 0
    for i, rec in enumerate(matched_records[:max_context_records], start=1):
        block = f"[Record {i} | id={rec['id']}]\n{rec['text']}\n"
        if total_chars + len(block) > max_context_chars:
            break
        parts.append(block)
        total_chars += len(block)
    return "\n\n".join(parts)


def generate_kg_answer(
    user_query: str,
    context: str,
    model: str = 'gpt-5.5',
    temperature: float = 0.2,
    client=None,
) -> str:
    """
    Non-streaming equivalent of _generate_response: same system prompt, grounding
    rules, and language-matching behaviour as the live KGSearchCultureBot.py, so the
    final answer text is directly comparable between the old bot and this notebook.
    """
    if client is None:
        client = make_openai_client()

    has_context = bool(context.strip())
    if has_context:
        grounding_rules = (
            "2. **Grounding**: Cite artifact URIs as links \u2014 e.g. ([\u03c4\u03af\u03c4\u03bb\u03bf\u03c2/title](URI)) \u2014 for every "
            "   distinct artifact you mention. Cite as many genuinely relevant sources as the evidence "
            "   supports; do not under-cite or settle for one or two links when more relevant artifacts "
            "   are available in the context.\n"
            "3. **Write as an expert, not a search report**: Never reference the retrieval process itself. "
            "   Do NOT use phrases like 'based on the retrieved context', 'according to the retrieved "
            "   objects', 'other retrieved items were not related to...', 'from the provided context', or "
            "   any similar meta-commentary about searching/retrieving. State facts directly, as something "
            "   you know. Never describe or list items that are NOT relevant \u2014 silently omit them instead; "
            "   only ever mention artifacts that are actually relevant to the question.\n"
            "4. **Depth and length**: Write a substantial, thorough answer \u2014 this is the default "
            "   expectation, not a fallback reserved for 'many matches'. Cover the relevant artifacts in "
            "   real detail (dates, places, materials, descriptions, subjects), draw connections between "
            "   them, and do not compress to a short list when the evidence supports more. Only write a "
            "   short answer when the evidence is genuinely thin \u2014 never pad, but always use what's there.\n"
            "5. **Analysis**: Do not just enumerate items \u2014 analyze them. Note patterns across the results "
            "   (shared periods, places, materials, types), explain significance or historical context, and "
            "   synthesize an interpretation rather than a raw catalogue.\n"
            "6. **Honesty**: If overall coverage of the topic is thin, you may say so briefly and in "
            "   general terms \u2014 but never list or describe specific unrelated items to illustrate it. "
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
        "   Greek \u2192 Greek, English \u2192 English. Never switch mid-answer.\n"
        + grounding_rules +
        "8. **Structure**: Organize with Markdown headings (grouping by place/period/theme as relevant to "
        "   the question), a short bullet list per artifact covering its key facts, and bold for key terms. "
        "   The answer should read as a clean, well-organized piece of writing.\n"
        "9. **Summary**: Always end with a short closing section \u2014 '## \u03a3\u03cd\u03bd\u03bf\u03c8\u03b7' for Greek answers, "
        "   '## Summary' for English \u2014 that synthesizes the key findings in a few sentences.\n"
        "10. **No follow-ups**: Do NOT end the response with follow-up question suggestions, "
        "    prompts the user could ask next, 'you might also want to know\u2026', 'further questions', "
        "    'feel free to ask\u2026', or any similar prompting hooks. Stop right after the summary.\n"
        "11. **Signature**: Begin every answer with a short identifying line, on its own, exactly: "
        "    '**\ud83c\udfdb\ufe0f CultureBot**' \u2014 then a blank line, then the answer itself."
    )

    user_prompt = (
        f"## Retrieved context\n\n{context}\n\n---\n\n"
        f"## Question\n{user_query}\n\n"
        f"*(Answer in the same language as the question above)*"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        # temperature=temperature,
    )
    return resp.choices[0].message.content


def ask_kg_full(
    user_query: str,
    driver,
    records_by_id: Dict[str, Dict[str, Any]],
    model: str = 'gpt-4o',
    limit: int = 100,
    temperature: float = 0.2,
    client=None,
) -> Dict[str, Any]:
    """
    End-to-end equivalent of KGSearchCultureBot.py's _ask_kg(): NL question -> Cypher ->
    Neo4j rows -> JSONL text lookup -> grounded final answer. Returns every intermediate
    artifact (cypher, rows, matched_records, context, answer) so each stage can be
    inspected/compared against the old bot's output for the same question.
    """
    cypher = generate_cypher_with_openai(user_query, model=model, limit=limit, client=client)
    rows = run_cypher(driver, cypher, max_rows=limit)
    item_ids = extract_item_ids(rows)
    matched_records = lookup_records_by_ids(records_by_id, item_ids)
    context = build_kg_context(matched_records, rows)
    answer = generate_kg_answer(user_query, context, model=model, temperature=temperature, client=client)
    return {
        'cypher': cypher,
        'rows': rows,
        'item_ids': item_ids,
        'matched_records': matched_records,
        'context': context,
        'answer': answer,
    }
