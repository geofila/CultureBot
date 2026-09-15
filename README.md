# SearchCultureBot

A chatbot that answers questions about **your own** cultural-heritage collection.

You give it a folder of documents — Markdown, PDF, JSONL or JSON, whatever you already
have — and, optionally, a Neo4j knowledge graph. It gives you a private ChatGPT-style web
page at `http://localhost:12012` where anyone you allow can ask questions in Greek or
English and get an answer built **only** from your records, with the source URI of every
item it used.

Under the hood it is [Open WebUI](https://github.com/open-webui/open-webui) (the chat
page) talking to an [Open WebUI Pipeline](https://github.com/open-webui/pipelines) (the
brain) that does hybrid search — FAISS semantic search + BM25 keyword search — and
optionally Neo4j, then asks an OpenAI model to write the answer.

---

## ⚠️ The two things you must supply yourself

This repository contains **code only**. It ships with **no data and no keys**. Nothing
will work until you provide both:

| You must provide | Where it goes | Step |
|---|---|---|
| 🔑 **Your own OpenAI API key** | the file `.env` in this folder (you create it) | [Step 3](#step-3--put-your-own-api-key-in-env) |
| 📚 **Your own dataset** | the folder `dataset/` (already here, empty) — any filenames | [Step 4](#step-4--put-your-own-data-in-dataset) |

Both are in `.gitignore`, so your key and your data can never be pushed to GitHub by
accident.

---

## Before you start — what you need

- A computer with **Linux, macOS or Windows** and about **15 GB of free disk space**
  (the pipeline image is large — it contains PyTorch).
- **Docker** + the **Docker Compose v2 plugin** → [Step 1](#step-1--install-docker).
- An **OpenAI API key** from <https://platform.openai.com/api-keys>. It must have
  credit on it — the bot pays per question and per indexed record.
- Your **dataset** — Markdown, PDF, JSONL or JSON files. See [Step 4](#step-4--put-your-own-data-in-dataset).
- *(Optional)* a running **Neo4j** database, if you also want graph queries. Without it
  the bot still works — you just run it in `rag` mode. See [Step 5](#step-5--optional--neo4j-knowledge-graph).
- No GPU is required with the default settings.

⏱️ Expect about **20–40 minutes** the first time, most of it Docker downloading images.

---

# Step-by-step installation

Follow the steps in order. After each one there is a ✅ check so you know it worked
before moving on.

## Step 1 — Install Docker

**Ubuntu / Debian:**

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker "$USER"
```

Now **log out and log back in** (or run `newgrp docker`), otherwise every `docker`
command will say "permission denied".

**macOS / Windows:** install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
and start it. It already includes Compose v2.

✅ **Check:** both commands print a version number.

```bash
docker --version
docker compose version
```

---

## Step 2 — Get the code and switch to the right branch

⚠️ **This is the step people get wrong.** The bot does **not** live on the default
branch. `git clone` gives you `main`, which contains only the project's website — no
`compose.yml`, no pipelines, nothing to run. You must check out **`culture_deploy`**.

```bash
git clone https://github.com/geofila/CultureBot.git
cd CultureBot
git checkout culture_deploy
```

Already cloned earlier and nothing looks familiar? You are on the wrong branch — just run:

```bash
git checkout culture_deploy
```

✅ **Check:** the first command prints `culture_deploy`, and the second lists
`compose.yml`, `.env.example`, `dataset/` and `sources/`.

```bash
git branch --show-current
ls
```

If `ls` shows only a `docs/` folder, the checkout did not happen — repeat it before
continuing.

> Every command from here on is run **from inside this folder** (`CultureBot/`), on the
> `culture_deploy` branch.

---

## Step 3 — Put your own API key in `.env`

`.env` is the single file that holds all your secrets. It does not exist yet — create it
by copying the template:

```bash
cp .env.example .env
```

You also need two random passwords. Run this twice and copy each result:

```bash
openssl rand -hex 32
```

Now open the file in an editor:

```bash
nano .env        # or: gedit .env / code .env / vim .env
```

Fill in these **three required** values:

| Variable in `.env` | Put this in it |
|---|---|
| `OPENAI_API_KEY` | Your own OpenAI key. It starts with `sk-`. Get it at <https://platform.openai.com/api-keys> |
| `WEBUI_SECRET_KEY` | The **first** random string from `openssl rand -hex 32` |
| `PIPELINES_API_KEY` | The **second** random string. You will paste this again later, in [Step 8](#step-8--connect-the-chat-page-to-the-bot) |

So the top of your `.env` should end up looking like this (with **your** values):

```ini
OPENAI_API_KEY=sk-proj-REPLACE-WITH-YOUR-REAL-KEY
OPENAI_MODEL=gpt-5.5
OPENAI_BASE_URL=https://api.openai.com/v1

WEBUI_SECRET_KEY=8f3c1d...your-first-random-string...
PIPELINES_API_KEY=a91be7...your-second-random-string...
```

Save and close the editor (in `nano`: `Ctrl+O`, `Enter`, `Ctrl+X`).

> **Which model will it use?** ⚠️ Read this — it is the single most common mistake.
> The `OPENAI_MODEL` line above is only used by the simpler `CultureBot` pipeline. The
> main `KGSearchCultureBot` pipeline **ignores it** and uses its own built-in default,
> `gpt-5.5`. If your OpenAI account cannot use that model, every answer will fail with a
> model error. You change it in the web admin panel *after* startup — see
> [Choosing the model and the mode](#choosing-the-model-and-the-mode). Only
> `OPENAI_API_KEY`, `NEO4J_URI`, `NEO4J_USER` and `NEO4J_PASSWORD` are read from `.env`
> by that pipeline.

✅ **Check:** the file exists and your key is in it, and git ignores it.

```bash
grep OPENAI_API_KEY .env      # shows your key
git status --short            # must NOT list .env
```

---

## Step 4 — Put your own data in `dataset/`

Copy your files into the `dataset/` folder that is already in this repo. **Filenames do
not matter** — the bot scans the whole folder (subfolders included) and handles each file
according to its type:

| If the file is… | …the bot does this with it |
|---|---|
| `.md` / `.markdown` | Indexes it for search. Each `##` section becomes one searchable chunk. |
| `.pdf` | Extracts the text page by page and indexes each page as a chunk. |
| `.jsonl` / `.ndjson` | Reads one `{"id": ..., "text": ...}` record per line, for the Knowledge-Graph lookup. |
| `.json` | Looks at its shape: a list of place records → your place taxonomy; a list/object of `{id, text}` → records; the portal's filter vocabulary → kept aside. |
| anything else | Ignored, and named in the startup log so you can see it was skipped. |

So this is enough:

```bash
cp ~/my-research/collection.md        dataset/
cp ~/my-research/catalogue.pdf        dataset/
cp ~/my-research/records-2024.jsonl   dataset/
```

Files named `*.example.*` and `dataset/README.md` are skipped on purpose — they are the
format samples shipped with this repo, not your data.

**Look at the samples** to see the structure that works best:

```bash
cat dataset/puretext_chunks.example.md
cat dataset/puretext_chunks.example.jsonl
cat dataset/kg_jsons/searchculture_places_taxonomy.example.json
```

### Markdown — the easiest way to start

One `##` heading per artifact. Everything under a heading becomes one searchable chunk,
and a `URI:` line lets the bot cite it (and lets graph results find their text):

```markdown
## Example marble head (Παράδειγμα κεφαλής)
- URI: https://example.org/collection/item/0001

**Type:** Sculpture (Γλυπτό)
**Material:** Marble (Μάρμαρο)
**Time Period:** 101 AD - 200 AD
**Location Found:** Example City

**Description:**
Free text describing the object. This is the text the bot searches and quotes.
```

### JSONL — for large collections and for graph mode

One object per line. The `id` **must be the same URI/id used in your Neo4j graph**, so a
graph result can be matched to its description:

```json
{"id": "https://example.org/collection/item/0001", "text": "**Title:** ...\n**Description:**\n..."}
```

### PDF — just drop it in

Each page becomes its own chunk, titled `<filename> — page N`. Scanned PDFs with no text
layer produce nothing (the log says so) — run OCR on them first.

> **Both a `.md` and a `.jsonl` holding the same records?** By default the bot indexes the
> Markdown and uses the JSONL only for graph lookups, so you are not charged to embed the
> same corpus twice. Change the `RAG_SOURCES` valve to `all` if you really want both.

✅ **Check:** your files are there, and git is ignoring them.

```bash
ls -lh dataset/
git status --short          # must NOT list your data files
```

You will see the bot confirm what it found, file by file, in the log at
[Step 6](#step-6--start-everything). For the full detail of every file type — accepted
field names, what makes a good chunk, how PDFs are handled — see
[`dataset/README.md`](dataset/README.md).

---

## Step 5 — *(Optional)* — Neo4j knowledge graph

Skip this if you do not have a Neo4j database — the bot works fine without it.

If you do, add your connection details to `.env`:

```ini
NEO4J_URI=neo4j://your-neo4j-host:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password
```

> `localhost` will **not** work here: the bot runs inside a container, where `localhost`
> means the container itself. Use the machine's IP, a hostname, or
> `host.docker.internal` (Docker Desktop) if Neo4j runs on your own machine.

Your graph must use the node labels and relationships the Cypher prompt was written
against (`Entity:ProvidedCHO`, `Entity:Place`, `Entity:TimeSpan`, `Entity:Concept`,
`Material`, with `LOCATED_IN`, `HAS_TEMPORAL_REFERENCE`/`FROM_PERIOD`, `HAS_TYPE`,
`HAS_SUBJECT`, `MADE_OF`). The full expected schema is at the top of
[`sources/pipelines/sc_kg_nl2cypher.py`](sources/pipelines/sc_kg_nl2cypher.py) — read it
before pointing the bot at a differently-shaped graph.

---

## Step 6 — Start everything

```bash
docker compose up -d
```

The **first run builds the pipeline image** (PyTorch + CUDA wheels, several GB). This can
take 10–30 minutes and only happens once. Watch it work:

```bash
docker compose logs -f pipelines
```

Press `Ctrl+C` to stop watching (the containers keep running).

✅ **Check:** both containers say `running` / `Up`.

```bash
docker compose ps
```

✅ **Check that the bot found your data.** The log prints one line per file:

```bash
docker compose logs pipelines | grep -A20 "Dataset scan"
```

```
Dataset scan — /app/pipelines/dataset: 1 markdown, 1 pdf, 1 records → ...
  • collection.md [markdown] — 482,190 chars, 1,204 ids
  • catalogue.pdf [pdf] — 88/90 pages with text
  • records-2024.jsonl [records] — 1,204 records
  • notes.txt [unsupported] — .txt not read
```

Anything you expected to see missing, or marked `unsupported` / `empty` / `unreadable`,
is the thing to fix before going on.

---

## Step 7 — Create your account

1. Open <http://localhost:12012> in a browser.
2. Click **Sign up** and create the first account.

> The **first account created becomes the administrator** automatically. Every account
> created after that stays `pending` until you approve it from the admin panel — this is
> deliberate, so a public deployment cannot be used by strangers.

✅ **Check:** you are logged in and see an empty chat page.

---

## Step 8 — Connect the chat page to the bot

The chat page and the bot are two separate services; you introduce them once.

1. Click your avatar (bottom-left) → **Admin Panel**.
2. Go to **Settings → Connections**.
3. Under **OpenAI API**, add a new connection (the **+** button):
   - **URL / Base URL:** `http://pipelines:9099`
   - **API Key:** the `PIPELINES_API_KEY` value from your `.env` (the second random
     string from Step 3)
4. **Save**.

> Use `http://pipelines:9099` exactly. This is the address *inside* the Docker network —
> `localhost:12011` will not work from the chat container.

✅ **Check:** open a new chat. The model dropdown now lists two bots:
**KGSearchCultureBot V2 (Clean)** (the main one) and **CultureBot** (the simpler RAG-only one).

---

## Step 9 — Ask your first question

1. Start a **New Chat** and select **KGSearchCultureBot V2 (Clean)** from the model list.
2. Ask something your dataset can answer, e.g. *"Show me marble sculptures from the Roman period"*.

The very first question prints `⏳ Initializing index…` and takes a few minutes: it is
embedding your whole corpus through the OpenAI API and building the FAISS + BM25 index.
This is cached, so later questions answer in seconds.

Watch it happen in another terminal:

```bash
docker compose logs -f pipelines
```

🎉 **That's it.** If you got a grounded answer with URIs, your installation is complete.

---

# Using the bot

## Choosing the model and the mode

These live in the web admin panel, **not** in `.env`:

**Admin Panel → Pipelines → select `KGSearchCultureBot`** → change the values → **Save**.

| Setting (valve) | Default | Change it to |
|---|---|---|
| `OPENAI_MODEL` | `gpt-5.5` | Any chat model your key can use (e.g. `gpt-4o`, `gpt-4o-mini`). **Set this if you get model errors.** |
| `QUERY_MODE` | `hybrid` | `rag` if you have no Neo4j (faster, avoids a wasted graph call) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | `text-embedding-3-large` (better, pricier), or `sentence-transformers/all-MiniLM-L6-v2` to embed locally on CPU for free |
| `TOP_K_SEMANTIC` / `TOP_K_BM25` | `100` | Lower for cheaper/faster answers |
| `TEMPERATURE` | `0.2` | Higher for more creative (less literal) answers |
| `RAG_SOURCES` | `auto` | `text` (index only `.md`/`.pdf`), `records` (only `.jsonl`/`.json`), or `all` (both, even if that duplicates a corpus) |
| `DATASET_DIR` | `/app/pipelines/dataset` | Only if you mount your data somewhere else |
| `EXTRA_FILES` | *(empty)* | Comma-separated paths to extra files outside the dataset folder |

> Changing `EMBEDDING_MODEL` or the retrieval settings invalidates the index cache — the
> next question rebuilds it (and re-pays for embeddings).
>
> Valve changes are stored inside the `pipelines` container, so check them again after
> any `docker compose build` or container recreate.

**The three query modes:**

| Mode | What it does | Needs Neo4j? |
|---|---|---|
| `hybrid` *(default)* | Graph query **and** hybrid search, merged into one grounded answer | Yes (falls back to RAG-only if the graph fails) |
| `kg` | Model writes Cypher → Neo4j → looks up each item's text in your records → answers | Yes |
| `rag` | FAISS + BM25 search over your indexed text only | No |

You can also override the mode per conversation by putting `[mode:rag]`, `[mode:kg]` or
`[mode:hybrid]` in the chat's system prompt.

## Everyday commands

| What you want | Command |
|---|---|
| Start | `docker compose up -d` |
| Stop | `docker compose down` |
| Restart just the bot | `docker compose restart pipelines` |
| See what is happening | `docker compose logs -f pipelines` |
| Check status | `docker compose ps` |
| After changing `.env` | `docker compose up -d` (recreates with new values) |
| After changing pipeline code or the Dockerfile | `docker compose build pipelines && docker compose up -d` |
| After adding/changing/removing dataset files | `docker compose restart pipelines` — new files are picked up and the index rebuilds by itself (the cache key covers every file in the folder) |

---

# Troubleshooting

**`error while interpreting services... set OPENAI_API_KEY in .env`**
Your `.env` is missing or that line is empty. Redo [Step 3](#step-3--put-your-own-api-key-in-env).
Compose refuses to start rather than run with an empty key.

**`❌ No data found` / `Nothing indexed for the RAG path`**
The bot lists everything it scanned in that same message and in the log. Check that your
files are in `dataset/` on the host, and that they are one of the types it reads
(`.md`, `.pdf`, `.jsonl`, `.json`). Confirm they reached the container with:

```bash
docker compose exec pipelines ls -R /app/pipelines/dataset
```

**One of my files was ignored**
Look for it in the `Dataset scan` log lines. The reason is always given:
- `unsupported` — not a type the bot reads (e.g. `.txt`, `.csv`, `.docx`). Convert it.
- `empty` — the file is 0 bytes.
- `unreadable` — a malformed JSON, or a scanned PDF with no text layer (OCR it first).
- not listed at all — its name contains `.example.`, or it is a `README.md`, a dotfile,
  or sits in a `cache/` folder. All of those are skipped by design.

**The answer is an OpenAI model error (`model_not_found`, `does not exist`, `400`)**
Your key cannot use the default `gpt-5.5`. Change the `OPENAI_MODEL` valve in
**Admin Panel → Pipelines** to a model you do have (editing `.env` does **not** fix this
for the main pipeline).

**No models in the dropdown / "no connection"**
Redo [Step 8](#step-8--connect-the-chat-page-to-the-bot). The URL must be
`http://pipelines:9099`, and the API key must match `PIPELINES_API_KEY` in `.env`
character for character.

**Answers are empty or say "no results"**
Ask something your dataset actually contains, then check how much got indexed in the log
(`Prepared N chunks for RAG from M source(s)`). If `N` is 1 for a large Markdown file,
it has no `##` headings, so the whole file became a single chunk — add headings.

**KG / hybrid mode never returns graph results**
Check `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`, that the URI is not `localhost`
(see [Step 5](#step-5--optional--neo4j-knowledge-graph)), and that the item ids in your
graph match the `id` values in your `.jsonl` records (or the `URI:` lines of your
Markdown sections).

**I want to force a full re-index**
The index rebuilds by itself whenever a dataset file changes, so you rarely need to. To
force it, delete the cache that `compose.yml` keeps on the host:

```bash
docker compose down
rm -rf data/pipelines-cache
docker compose up -d
```

**`no configuration file provided: not found` / there is no `compose.yml`**
You are on the wrong branch. The code lives on `culture_deploy`, not on `main` (which
holds only the project website). Run `git checkout culture_deploy` — see
[Step 2](#step-2--get-the-code-and-switch-to-the-right-branch).

**`permission denied while trying to connect to the Docker daemon`**
You skipped the log-out after `usermod -aG docker`. Run `newgrp docker` or log out and in.

**`port is already allocated`**
Something else uses 12012 or 12011. In `compose.yml`, change the **left** side of the
mapping, e.g. `"127.0.0.1:13012:8080"`.

---

# Reference

## How your key and your data actually reach the bot

```
.env  ──read by──▶  compose.yml  ──injected as env vars──▶  container
                                                              │
                                         os.getenv("OPENAI_API_KEY")
                                                              ▼
./dataset  ──mounted read-only──▶  /app/pipelines/dataset  ──scanned by──▶  the pipeline
                                          │
             dataset_loader.py sorts each file by type: .md/.pdf → search index,
             .jsonl/.json → records + place taxonomy
```

No key is ever written into the source code, and no dataset is baked into the image.
Both stay on your machine.

## Architecture

```
┌─────────────┐        ┌──────────────────────────────┐        ┌────────────────┐
│  Open WebUI │──API──▶│  Pipelines (RAG + KG)        │──API──▶│  OpenAI        │
│  (chat page)│        │  KGSearchCultureBot_v2.py    │        │  (LLM + embed) │
└─────────────┘        │  FAISS + BM25 + Cypher       │        └────────────────┘
      ▲                └──────────────┬───────────────┘
      │                               │ (kg / hybrid modes only)
   browser                            ▼
                              ┌────────────────┐
                              │  Neo4j graph   │  (optional)
                              └────────────────┘
```

| Service | Image | Address on your machine | Purpose |
|---|---|---|---|
| `open-webui` | `ghcr.io/open-webui/open-webui:main` | `http://localhost:12012` | Chat page + user accounts |
| `pipelines` | built from `sources/Dockerfile.pipelines` | `http://localhost:12011` | The RAG/KG brain |

Both ports are bound to `127.0.0.1`, so nothing is exposed to your network by default.
`compose.yml` also contains two optional services, commented out: a local **Ollama**
backend and an internal **auto-redeploy** webhook.

## Repository layout

```
.
├── compose.yml                     # The stack. Mounts ./dataset into the pipeline
├── .env.example                    # Template for your secrets — copy to .env (Step 3)
├── .gitignore                      # Keeps your data & secrets out of git
├── data/                           # Chat history, accounts, index cache (gitignored)
├── dataset/                        # 👉 YOUR DATA GOES HERE — any filenames (gitignored)
│   ├── README.md                   # What each file type is used for
│   ├── *.example.*                 # Small samples showing the formats
│   └── kg_jsons/                   # Sample place taxonomy + filter vocabulary
└── sources/
    ├── build.sh                    # Optional: build the image by hand
    ├── Dockerfile.pipelines        # How the pipelines image is built
    └── pipelines/
        ├── KGSearchCultureBot_v2.py    # Main pipeline (KG + hybrid RAG)
        ├── SearchCultureBot.py         # Simpler RAG-only pipeline
        ├── dataset_loader.py           # Scans dataset/ and sorts files by type
        └── sc_kg_nl2cypher.py          # Natural language → Cypher + graph schema
```

## Every `.env` variable

| Variable | Required | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | **Your own OpenAI key** — used for answers *and* embeddings |
| `WEBUI_SECRET_KEY` | ✅ | Random secret signing login sessions (`openssl rand -hex 32`) |
| `PIPELINES_API_KEY` | ✅ | Random token the chat page uses to call the bot (`openssl rand -hex 32`) |
| `OPENAI_MODEL` | – | Used by the `CultureBot` pipeline only; set the main bot's model in the admin panel |
| `OPENAI_BASE_URL` | – | Only for Azure OpenAI or an OpenAI-compatible proxy |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | for `kg`/`hybrid` | Your graph database connection |
| `WEBUI_AUTH`, `ENABLE_SIGNUP`, `DEFAULT_USER_ROLE` | – | Login / signup behaviour (safe defaults already set) |
| `WEBUI_SESSION_COOKIE_SECURE`, `WEBUI_SESSION_COOKIE_SAME_SITE` | – | Set `SECURE=true` when serving over HTTPS |
| `RATE_LIMIT_ENABLED`, `RATE_LIMIT_REQUESTS_PER_MINUTE` | – | Throttling |
| `ENABLE_COMMUNITY_SHARING`, `SAFE_MODE` | – | Open WebUI feature switches |
| `OLLAMA_BASE_URL` | – | Only if you enable the optional `ollama` service |
| `DEPLOY_AUTH_TOKEN` | – | Only if you enable the optional `deploy` service |

## A note on cost

Every question costs OpenAI credit, and the **first** question costs more: it embeds your
entire corpus once. A large corpus with `text-embedding-3-large` can be significant — start
with `text-embedding-3-small` (the default), or switch `EMBEDDING_MODEL` to
`sentence-transformers/all-MiniLM-L6-v2` to embed locally for free. Keep the index cache
(see Troubleshooting) so you only pay for embedding once.

## Security

- **Never commit `.env` or your real `dataset/` files.** Both are gitignored — run
  `git status` before your first push to be sure.
- **If a key was ever pasted into a file that got committed, rotate it.** Deleting it in a
  later commit does not remove it from git history.
- Ports are bound to `127.0.0.1`. To publish the bot, put a reverse proxy (Caddy, Nginx)
  with TLS in front and set `WEBUI_SESSION_COOKIE_SECURE=true`.
- Keep `DEFAULT_USER_ROLE=pending` so new sign-ups need your approval.
- The Cypher the model generates is validated as read-only before it runs, but give the
  bot a **read-only Neo4j user** anyway.
