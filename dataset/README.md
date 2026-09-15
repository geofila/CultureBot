# `dataset/` — your data goes here

Drop your files in this folder. **Filenames do not matter** and subfolders are fine: on
startup the pipeline scans the whole folder and decides what each file is from its type
(see [`sources/pipelines/dataset_loader.py`](../sources/pipelines/dataset_loader.py)).

`compose.yml` already mounts this folder read-only at `/app/pipelines/dataset` — there is
nothing to uncomment and nothing to rename.

The real dataset this project was built on is confidential and is **not** included here.
Everything you add is gitignored; only this README and the `*.example.*` samples are
tracked.

## What happens to each file type

| Extension | Read as | Used for |
|---|---|---|
| `.md`, `.markdown` | Text corpus, split on `#`/`##`/`###` headings | The FAISS + BM25 search index (`rag` and `hybrid` modes) |
| `.pdf` | Text extracted per page, each page turned into a `## <file> — page N` section | The same search index |
| `.jsonl`, `.ndjson` | One record per line | `{id → text}` lookup for the Knowledge-Graph path (`kg` and `hybrid` modes) |
| `.json` | Sniffed by shape — see below | Place taxonomy, records, or filter vocabulary |
| anything else | Ignored | Reported in the startup log so you can see it was skipped |

Skipped on purpose: files whose name contains `.example.`, any `README.md`, dotfiles, and
anything under a `cache/` folder.

## Markdown

One artifact per `##` section. Keep a `URI:` line in each section: it is what the answer
cites, and it also registers the section as a record, so the Knowledge-Graph path can
quote your text even when you supplied no JSONL at all.

```markdown
## Example marble head (Παράδειγμα κεφαλής)
- URI: https://example.org/collection/item/0001

**Type:** Sculpture (Γλυπτό)
**Material:** Marble (Μάρμαρο)

**Description:**
Free text describing the object.
```

A Markdown file with no headings becomes a single chunk, which retrieves poorly — split
it up. See [`puretext_chunks.example.md`](puretext_chunks.example.md).

## JSONL / NDJSON

One JSON object per line:

```json
{"id": "https://example.org/collection/item/0001", "text": "..."}
```

`id` may also be spelled `uri`, `url` or `identifier`; `text` may be `content`,
`chunk_text`, `chunk`, `body` or `description`. A line with text but no id still gets
indexed (under a generated id); unparseable lines are skipped and counted in the log.

**The `id` must match the item id in your Neo4j graph** (`ProvidedCHO.id`). After a Cypher
query returns item ids, the pipeline looks up their text here — a mismatch means the graph
finds items but the answer has nothing to quote. See
[`puretext_chunks.example.jsonl`](puretext_chunks.example.jsonl).

## JSON

The shape decides the role:

| Shape | Treated as |
|---|---|
| List of objects with `path`, `path_text`, `label_el`/`label_en` | **Place taxonomy** — expands a place named in a question into its whole branch before the Cypher is written ([sample](kg_jsons/searchculture_places_taxonomy.example.json)) |
| List of objects with an id + text field | **Records**, exactly like JSONL |
| Object mapping `id → text` (or `id → {text: ...}`) | **Records** |
| Object with `advanced_search_controls` | **Filter vocabulary** — kept aside; the current pipeline does not use it ([sample](kg_jsons/searchculture_filters_from_page.example.json)) |
| An empty list `[]` | An empty place taxonomy — the harmless placeholder |
| Anything else | Ignored, and reported in the log |

No place taxonomy is fine: the Cypher prompt simply gets no place candidates.

## PDF

Text is extracted page by page; each page becomes its own chunk titled
`<filename> — page N`. A scanned PDF with no text layer yields nothing and says so in the
log — run OCR on it first.

## Which files feed the search index

By default (`RAG_SOURCES=auto`) the index is built from your Markdown and PDF text, and
JSON/JSONL records are used only for Knowledge-Graph lookups. If you supply **only**
records, they are indexed instead, so a JSONL-only dataset still works.

That default exists because a `.md` and a `.jsonl` of the same collection are usually two
views of one corpus, and indexing both would embed everything twice. Set the `RAG_SOURCES`
valve (**Admin Panel → Pipelines**) to `all` if you want exactly that, or to `text` /
`records` to pin one side.

## Checking it worked

The startup log lists every file and what became of it:

```bash
docker compose logs pipelines | grep -A20 "Dataset scan"
```

```
Dataset scan — /app/pipelines/dataset: 1 markdown, 1 pdf, 1 records → 2 text source(s), 1204 record(s), 0 place(s)
  • collection.md [markdown] — 482,190 chars, 1,204 ids
  • catalogue.pdf [pdf] — 88/90 pages with text
  • records-2024.jsonl [records] — 1,204 records
  • notes.txt [unsupported] — .txt not read
```

Adding, editing or removing a file changes the cache key, so the index rebuilds by itself
on the next restart.
