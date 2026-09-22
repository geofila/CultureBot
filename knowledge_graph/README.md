# Building a CultureBot graph from EDM-style data

This directory contains a practical route from **compact JSON-LD with Europeana
Data Model (EDM) terms** to the Neo4j property graph queried by CultureBot. The
graph makes object, place, period, subject, type, provider, and image metadata
available for structured search. EDM is the metadata model; JSON-LD is one
way to serialize it. This importer expects the compact SearchCulture-style
serialization shown below. It is a selective projection of the source
metadata, not a lossless EDM or RDF store.

The demonstration described in the [CultureBot paper](../README.md#paper)
reports 81,852 SearchCulture records. The scripts here reflect that input shape and the graph
schema expected by the current [Cypher prompt](../sources/pipelines/sc_kg_nl2cypher.py).
Test mappings and query results on a sample of **your** collection before
importing a full export.

## Choose an import path

| Path | Script | Result | Use it when |
|---|---|---|---|
| CultureBot property graph | [`scripts/add_data_to_neo4j_extended_v2.py`](scripts/add_data_to_neo4j_extended_v2.py) | `:Entity:ProvidedCHO` and related nodes with the relationship names CultureBot expects | You want the `kg` or `hybrid` pipeline mode. **Start here.** |
| Optional authority enrichment | [`scripts/enrich_hierarchy.py`](scripts/enrich_hierarchy.py) | Adds parent links and labels by fetching `semantics.gr` authority records | Your data uses those authority URIs and needs fuller place or concept hierarchies. |
| RDF import | [`scripts/add_data_to_neo4j_rdf.py`](scripts/add_data_to_neo4j_rdf.py) | An n10s RDF graph, usually with `:Resource` nodes | You want to explore more of the source RDF/JSON-LD model in a **separate** database. This is not a drop-in input for CultureBot's current Cypher prompt. |

[`one_sample.json`](one_sample.json) is a single, formatted SearchCulture-style
JSON-LD document for inspecting the expected structure. It is not already
JSONL, so the tutorial converts it before import. No complete production
collection or Neo4j database is included here.

## Input contract

The CultureBot importer reads a UTF-8 `.jsonl` file: **one complete JSON object
per line**, with an `@graph` array on each line. Each graph entry needs a stable
`@id` and a compact `@type`, for example `edm:ProvidedCHO`, `ore:Aggregation`,
`edm:Place`, `edm:TimeSpan`, or `skos:Concept`. A record may contain several
entries, as the supplied sample does.

```json
{"@graph":[{"@id":"https://example.org/object/1","@type":"edm:ProvidedCHO","title":{"@language":"en","@value":"Marble vessel"},"spatial":"https://example.org/place/1"},{"@id":"https://example.org/place/1","@type":"edm:Place","prefLabel":{"@language":"en","@value":"Athens"}}]}
```

The importer reads **compact property keys**, including `subject`, `type`,
`spatial`, `temporal`, `aggregatedCHO`, `medium`, and `dc:type`. It does not run
JSON-LD expansion or interpret arbitrary `@context` mappings. An export with
full predicate IRIs, different aliases, nested record envelopes, or `@id`
values without HTTP(S) links needs a conversion or mapping update first.
Relationship targets must be HTTP(S) strings or objects containing an HTTP(S)
`@id`. Literal text can be a string, an `{"@value": ...}` object, or a list
of these. Blank lines are skipped; malformed JSON stops the import.

### What the property graph stores

Every imported `@id` becomes an `:Entity` node with an `id` and a display
`label`. The first `@type` determines its second Neo4j label. Repeated IDs are
merged. Referenced IDs absent from the current `@graph` become placeholder
`:Entity` nodes; they gain a type and properties only if a later record
defines them.

| Source entry | Properties retained on the node |
|---|---|
| `edm:ProvidedCHO` | Flattened title and description, English and Greek title, created date text and parsed year range, extent, medium, literal `dc:type`, source, identifier, `edm:type`, literal `dcterms:spatial`. |
| `edm:Place` | Flattened label; latitude and longitude as supplied, usually strings. |
| `edm:TimeSpan` | Flattened label and integer `begin_year`/`end_year` when parsing succeeds. |
| `skos:Concept` and other types | Stable ID and flattened display label; the first `@type` supplies the extra label. |
| `ore:Aggregation` | Provider names, rights, primary image URL, landing page, object/thumbnail URL, and `hasView` URLs. |
| Literal `medium` values | Separate `:Material {name}` nodes; translations can therefore become separate nodes. |
| External authority URIs | `:External {id}` nodes reached from `exactMatch` or `sameAs`. |

| Compact key | Neo4j edge | Typical direction |
|---|---|---|
| `subject`, `type` | `HAS_SUBJECT`, `HAS_TYPE` | object to concept |
| `spatial`, `temporal` | `LOCATED_IN`, `HAS_TEMPORAL_REFERENCE` | object to place or time span |
| `aggregatedCHO` | `AGGREGATES` | aggregation to object |
| `broader`, `narrower`, `broadMatch` | `BROADER_THAN`, `NARROWER_THAN`, `BROAD_MATCH` | concept to concept |
| `isPartOf` | `IS_PART_OF` | place to parent place, when used that way in the source |
| `exactMatch`, `sameAs` | `EXACT_MATCH`, `SAME_AS` | entity to external URI |
| Literal `medium` | `MADE_OF` | object to material |

The script also rebuilds `FROM_PERIOD` edges **after** import. These are
inferred from overlap between a `ProvidedCHO`'s parsed `created` year range
and *every* `TimeSpan` with numeric bounds. They are distinct from
source-declared `HAS_TEMPORAL_REFERENCE` edges. Overlap is not proof that the
source cataloguer assigned a particular historical period; broad or approximate
dates can yield many matches.

## Small tutorial: import the supplied sample

These commands are run from the repository root. Use a dedicated Neo4j database
for the tutorial. The importer creates constraints and indexes and writes to the
database, even when `RESET_DB=false`.

1. Start Neo4j 5.24 or newer, or use an existing instance. For a local Docker
   example, Neo4j Browser is at <http://localhost:7474> and Bolt is on port
   `7687`:

   ```sh
   docker run --name culturebot-neo4j -d -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 -v culturebot-neo4j-data:/data neo4j:5.26
   ```

   On a fresh container, sign in to Neo4j Browser as `neo4j` with its initial
   password `neo4j`, then set a new password. The volume preserves data across
   container restarts. The importer uses dynamic labels, which require Neo4j
   5.24 or later ([Cypher documentation](https://neo4j.com/docs/cypher-manual/5/clauses/set/)).

2. Install Python 3.10+ dependencies in an environment of your choice:

   ```sh
   python -m pip install neo4j python-dotenv tqdm requests
   ```

3. Make the formatted sample into one JSONL line. The output directory is
   ignored by Git:

   ```sh
   python -c "import json; from pathlib import Path; s=Path('knowledge_graph/one_sample.json'); d=Path('knowledge_graph/data'); d.mkdir(exist_ok=True); (d/'sample.jsonl').write_text(json.dumps(json.loads(s.read_text(encoding='utf-8')), ensure_ascii=False)+'\n', encoding='utf-8')"
   ```

4. Copy the repository's `.env.example` to `.env` if you have not already done
   so. Set these values in `.env`; use the password you just set:

   ```dotenv
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=replace-with-your-password
   JSONL_PATH=knowledge_graph/data/sample.jsonl
   START_LINE=0
   RESET_DB=false
   ```

   `.env` is ignored by Git. The importer loads it through `python-dotenv`.
   `JSONL_PATH` is resolved from the **current working directory**, which is
   why the commands here start at the repository root.

5. Run the property-graph importer:

   ```sh
   python knowledge_graph/scripts/add_data_to_neo4j_extended_v2.py
   ```

6. In Neo4j Browser, check that the object and its connections were created:

   ```cypher
   MATCH (item:Entity:ProvidedCHO)
   RETURN item.id AS item_id, item.title AS title,
          item.created_start_year AS from_year,
          item.created_end_year AS to_year
   LIMIT 10;
   ```

   ```cypher
   MATCH (item:Entity:ProvidedCHO)-[rel]-(related)
   RETURN type(rel) AS relationship, labels(related) AS labels,
          related.id AS id, related.label AS label
   LIMIT 25;
   ```

The sample should yield one `ProvidedCHO`, one linked `Aggregation`, and links
to its subjects, types, place, and declared time span. `Marble` and `Μάρμαρο`
will appear as separate material nodes. You can repeat the import because
nodes and relationships are merged, but deleted or changed source links are
**not** automatically removed from an existing graph.

## Import a collection

Convert your EDM-style export to the JSONL contract above. Keep each source
`@id` stable across runs and preserve the aggregation's `aggregatedCHO` target
so it equals the corresponding object's `@id`. Put the resulting file under
`knowledge_graph/data/`, then point `JSONL_PATH` at it and run the same script.
For a larger collection, sample and validate a few lines before the full run.

`START_LINE` is a **zero-based count of input lines to skip**. It can resume
after an interrupted run, but the script has no automatic checkpoint: record
the last completed line yourself. `RESET_DB=true` deletes all nodes and
relationships in the connected database before ingestion; use it only for a
database you intend to empty. Rebuilding a clean graph is the safest way to
remove relationships or nodes that disappeared from the source. After every
run, the script deletes and regenerates all `FROM_PERIOD` edges, which can be
slow on a large graph. Do not run concurrent imports into the same database.

Check coverage with queries such as:

```cypher
MATCH (item:Entity:ProvidedCHO)
WITH item,
     EXISTS { (item)-[:LOCATED_IN]->() } AS has_place,
     EXISTS { (item)-[:HAS_TEMPORAL_REFERENCE]->() } AS has_declared_period
RETURN count(item) AS objects,
       sum(CASE WHEN has_place THEN 1 ELSE 0 END) AS objects_with_place,
       sum(CASE WHEN has_declared_period THEN 1 ELSE 0 END) AS objects_with_declared_period;
```

```cypher
MATCH (n:Entity)
WHERE size(labels(n)) = 1
RETURN n.id AS unresolved_reference
LIMIT 25;
```

The second query finds placeholder nodes that never received a more specific
type; some may be legitimate external references. Also inspect a few source
records beside their graph neighbourhoods and compare titles, place roles,
periods, rights, and image links.

## Use the graph in CultureBot

The graph import alone is not the complete retrieval dataset. The pipeline
expects a separate `.jsonl` or `.json` file under [`dataset/`](../dataset/README.md)
whose records have `{ "id": "...", "text": "..." }`. Each `id` must exactly
match a `ProvidedCHO.id` in Neo4j. The `text` should render the object's title,
description, useful metadata, and source URI so CultureBot can explain and
link its graph-selected records. For the supplied sample, a minimal companion
record would be:

```json
{"id":"https://www.searchculture.gr/aggregator/edm/mnam/000150-688002","text":"Title: Head of statue of Satyr. Description: Idealized head of a young satyr, a Roman copy of the second century CE. Source: https://www.searchculture.gr/aggregator/edm/mnam/000150-688002"}
```

That example is illustrative: build the text from your own source records and
check source rights before release. The ingestion scripts do **not** generate
these text records. Markdown and PDF collection text can also feed the FAISS
and BM25 retrieval path.

Set `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` in the repository `.env`,
then follow the [main setup guide](../README.md#use-a-neo4j-knowledge-graph).
If Neo4j runs on the host and CultureBot runs in Docker Desktop, change the
pipeline's URI from `bolt://localhost:7687` to
`neo4j://host.docker.internal:7687`; inside a container, `localhost` refers
to that container. Restart the pipeline after changing its environment.
Select `kg` or `hybrid` in the main pipeline's `QUERY_MODE` valve.

As described in the paper, CultureBot grounds candidate entities, asks a
language model for a Cypher query, checks the query against read-only rules
and Neo4j `EXPLAIN`, retrieves matching records, and combines graph metadata
with source text. In `hybrid` mode it also uses lexical and semantic text
retrieval before composing a source-linked answer. Give the runtime Neo4j
account **read-only** privileges where your Neo4j edition supports them; the
importer needs write and schema privileges. On a single-user edition, keep
the runtime database isolated from other workloads. If you use separate
importer and runtime accounts, update `.env` with the runtime credentials
after ingestion and recreate the pipeline container to apply them.

## Optional: enrich SearchCulture authority hierarchies

`enrich_hierarchy.py` follows `semantics.gr/authorities/` URIs upward through
`isPartOf`, `broader`, and `broadMatch`. It fetches authority records over HTTP,
caches responses, and merges parent nodes and edges. This step is
**SearchCulture-specific**; do not run it on unrelated vocabularies without
adapting the authority prefix and response parser. It can add spatial or
concept hierarchy beyond what was present in the original export, but a
remote authority may be unavailable or incomplete.

```sh
python knowledge_graph/scripts/enrich_hierarchy.py --dry-run --max-nodes 20 --debug --cache knowledge_graph/data/authority_cache.json
python knowledge_graph/scripts/enrich_hierarchy.py --only-frontier --cache knowledge_graph/data/authority_cache.json
```

The dry run still fetches and caches remote responses; it only suppresses
graph writes. The script reads Neo4j settings from `.env` or `--uri`/`--user`
and `NEO4J_PASSWORD`. Use `--sleep` to reduce request rate and `--max-nodes`
to bound an exploratory run. Cached failures may need a fresh cache file after
the remote service becomes available again.

## Alternative: import RDF with n10s

The RDF importer uses Neo4j's **neosemantics (n10s)** plugin and parses the
JSON-LD `@context` and predicates more generally. Install a plugin build
compatible with your Neo4j version and use a separate instance or database.
This route creates an RDF-oriented graph whose labels and relationship names are not the
`Entity`/`ProvidedCHO` schema expected by the current CultureBot prompt.
It is useful for inspecting source semantics, not as an automatic replacement
for the property-graph import above.

After installing n10s, export `NEO4J_PASSWORD` in your shell and run:

```sh
python knowledge_graph/scripts/add_data_to_neo4j_rdf.py --uri bolt://localhost:7687 --user neo4j --input knowledge_graph/data/sample.jsonl --checkpoint knowledge_graph/data/rdf-checkpoint.json --failed-log knowledge_graph/data/rdf-failed-lines.jsonl --batch-size 50
```

This script checks for the n10s procedure, creates its URI constraint, and
initializes graph configuration if absent. It retries failed batches, falls
back to individual records, and writes a checkpoint and failed-record log.
The failed log includes original payloads, so keep it private. Do not reuse a
checkpoint with a different input file. A failed batch may have written some
triples before it is retried, so verify counts and inspect failures on a test
instance before relying on a large import. See the [n10s documentation](https://neo4j.com/labs/neosemantics/)
for plugin installation and graph configuration.

## Known limits and adaptation points

- The property importer retains selected fields only. It does not preserve
  `@context`, all EDM classes and properties, full multilingual structures,
  provenance detail, or all media relations. A `WebResource` or `Agent` entry
  may become a node, but there is no general mapping from creators or digital
  resources to objects.
- It flattens most multilingual values into a slash-separated string. Greek
  and English material literals are not automatically reconciled.
- `Place.lat` and `.long` are not converted to numbers. Created dates and time
  span bounds are parsed heuristically; inspect BCE dates, uncertain ranges,
  centuries, and localized date strings before using period filters.
- Missing target records produce untyped placeholders. A graph with many of
  these can make queries that require `:Place`, `:Concept`, or `:TimeSpan`
  miss results.
- `FROM_PERIOD` is based on date overlap and can connect objects to many
  periods. For some collections you may want only source-declared
  `HAS_TEMPORAL_REFERENCE` or a narrower inference rule.
- The Cypher prompt is tailored to the demonstrated schema. If you change
  labels, property names, relationship roles, or aggregation layout, update
  [`sc_kg_nl2cypher.py`](../sources/pipelines/sc_kg_nl2cypher.py) and test its
  example queries against your graph.

For background on the EDM distinction between cultural objects,
aggregations, and digital representations, see [Europeana's EDM documentation](https://pro.europeana.eu/index.php/page/edm-documentation).
The ingestion here deliberately models only the subset needed for the current
CultureBot retrieval workflow.
