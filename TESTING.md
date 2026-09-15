# Testing the bot with the sample dataset

The `dataset/` folder currently holds a small **fake** collection — the "Fake Museum of
Example Antiquities" — so you can check the whole pipeline end to end before you load your
real data. All of it is gitignored, so none of it will be pushed.

| File | Type | What is in it |
|---|---|---|
| `fake_collection.md` | Markdown | 6 catalogued objects, one per `##` section, each with a `URI:` line (items 0001–0006) |
| `excavation_report_2024.pdf` | PDF, 3 pages | An excavation report: Trench A mosaic, Trench B cistern, conservation notes |
| `fake_records.jsonl` | JSONL | 4 **more** objects that are *not* in the Markdown (items 0007–0010) |
| `fake_places.json` | JSON | A place taxonomy for the 8 findspots used above |

Deliberately, the JSONL items are different objects from the Markdown ones. That is what
lets you see the `RAG_SOURCES` valve working: by default only the Markdown and PDF are
searched, and the JSONL is used for Knowledge-Graph lookups.

## Load it

```bash
docker compose restart pipelines
sleep 20
docker compose logs pipelines | grep -A5 "Dataset scan"
```

Each of the two pipelines scans on startup, so you get this block twice — once per
pipeline:

```
INFO:KGSearchCultureBot_v2:Dataset scan — /app/pipelines/dataset: 1 markdown, 1 pdf, 1 places, 1 records → 2 text source(s), 10 record(s), 8 place(s)
INFO:KGSearchCultureBot_v2:  • excavation_report_2024.pdf [pdf] — 3/3 pages with text
INFO:KGSearchCultureBot_v2:  • fake_collection.md [markdown] — 5,022 chars, 6 ids
INFO:KGSearchCultureBot_v2:  • fake_places.json [places] — 8 places
INFO:KGSearchCultureBot_v2:  • fake_records.jsonl [records] — 4 records
```

> **No `Dataset scan` lines at all?** Either the container is still starting (wait ~20 s
> and look again), or it is running an image built before the folder-scanning change.
> Rebuild it: `docker compose build pipelines && docker compose up -d pipelines`.

The index itself is built lazily, on your **first question** — not at startup. After you
ask one, the log adds:

```
Prepared 9 chunks for RAG from 2 source(s)
```

9 chunks = 6 Markdown sections + 3 PDF pages. That line confirms indexing worked.

Set `QUERY_MODE` to `rag` in **Admin Panel → Pipelines → KGSearchCultureBot** for the
tests below unless a test says otherwise — that way you are testing retrieval without
needing Neo4j.

---

## Questions to ask

### 1. Basic retrieval from the Markdown

| Ask | A correct answer contains |
|---|---|
| Which objects in the collection are made of marble? | Only the **marble head of a youth** (FME-0001) — Pentelic marble, Roman, 101–200 AD, found at Ancient Messene |
| Show me everything from the Classical period | Three items: bronze **strigil** (0002), red-figure **kylix** (0004), gold **stater** of Philip II (0006) |
| Which item has a carnation and tulip design? | The **Ottoman glazed tile** (0005) from Ioannina, cobalt blue and turquoise |
| What can you tell me about the red-figure kylix? | Athens, 480–450 BC, symposiast with a lyre in the medallion, five draped youths outside, restored from 11 fragments, circle of the Brygos Painter |

### 2. Greek queries (tests the Greek-aware BM25 tokenizer)

| Ask | A correct answer contains |
|---|---|
| Τι αντικείμενα έχεις από τη Θεσσαλονίκη; | The **φορητή εικόνα Αγίου Δημητρίου** (0003), 18th century |
| Πόσο ψηλή είναι η μαρμάρινη κεφαλή; | **24 cm** (and width 17.5 cm) |
| Ποια αντικείμενα είναι από την Ήπειρο ή τα Ιωάννινα; | The Ottoman tile (0005). The iconostasis fragment (0009) only appears if records are indexed — see test 5 |

### 3. The PDF (each page is its own chunk)

| Ask | A correct answer contains |
|---|---|
| What was found in Trench B? | A **Byzantine cistern**: rubble masonry with lime mortar and crushed brick, hydraulic plaster preserved to 1.8 m, cooking ware, iron nails, **three gold coins of Justinian I**, backfilled in the late 6th century |
| Describe the mosaic floor and how it was dated. | Two **dolphins flanking a trident**, black/white/ochre tesserae, dated to the **2nd century AD** from pottery in the bedding layer; 12 border fragments lifted |
| What are the storeroom temperature and humidity? | **20 °C and 55% RH**, with two excursions above 65% in August 2024 |

### 4. Cross-source synthesis (the real test of `hybrid`)

| Ask | A correct answer contains |
|---|---|
| What conservation treatment did the marble head receive, and where exactly was it found? | Treatment from the PDF (**5% acrylic resin in acetone** at the neck fracture; the burnt cheek deliberately left untreated) **plus** the findspot (destruction layer in Trench A, **40 cm above the mosaic**, Ancient Messene) — it has to combine the PDF and the catalogue entry |
| Which items are associated with the Byzantine period? | The **solidus of Justinian I** (0008, records only) and the **cistern** in Trench B (PDF) |

### 5. Records-only items — tests the `RAG_SOURCES` valve

Ask this first with the default `RAG_SOURCES=auto`:

> Tell me about the Mycenaean stirrup jar.

Expected: **no results**. The stirrup jar lives only in `fake_records.jsonl`, and `auto`
indexes the Markdown and PDF for search. That is the intended behaviour, not a bug.

Now set `RAG_SOURCES` to `all` in the admin panel, save, and ask again:

> Tell me about the Mycenaean stirrup jar.

Expected: Mycenae, 1300–1200 BC, false spout and three handles, stylised **octopus
tentacles**, repaired handle. The index rebuilds once when you change the valve.

| Ask (with `RAG_SOURCES=all`) | A correct answer contains |
|---|---|
| How many coins are in the collection and what are they? | **Two**: the gold stater of Philip II (Pella) and the gold solidus of Justinian I (Thessaloniki). With `auto` it will find only the stater |
| What glass objects do you have? | The **Roman glass unguentarium** (0010) from Patras, pale green, iridescent weathering |

### 6. Grounding check — it must refuse to invent

| Ask | A correct answer |
|---|---|
| Do you have any Egyptian mummies? | Says there is nothing in the dataset about Egyptian mummies. **If it invents one, grounding is broken.** |
| Έχεις τεκμήρια από την Αίγυπτο; | Same, in Greek |
| Who directed the 2024 excavation? | Says the documents do not name a director (they genuinely do not) |

### 7. Only if you connected Neo4j (`kg` / `hybrid` mode)

These need a real graph whose item ids match the `https://fake.example.org/item/NNNN`
URIs, so they will not work on the sample data alone. Once you have one:

> φέρε τεκμήρια από την Πελοπόννησο

The place taxonomy in `fake_places.json` puts Ancient Messene, Olympia and Mycenae under
Πελοπόννησος, so a correct Cypher query expands the region into those three findspots
rather than matching the word "Πελοπόννησος" in the text (which appears nowhere in it).

---

## When you are done

Remove the fake data before loading your own, so the two do not mix:

```bash
rm dataset/fake_collection.md dataset/fake_records.jsonl \
   dataset/fake_places.json dataset/excavation_report_2024.pdf
docker compose restart pipelines
```

The index rebuilds automatically, because the cache key covers every file in the folder.
