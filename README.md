# SearchCultureBot

A Retrieval-Augmented Generation (RAG) chatbot for cultural-heritage data, built on
[Open WebUI](https://github.com/open-webui/open-webui) and its
[Pipelines](https://github.com/open-webui/pipelines) framework. The bot answers
questions about movable monuments and cultural-heritage records by combining:

- a **hybrid RAG index** (FAISS semantic search + BM25 keyword search with Reciprocal
  Rank Fusion) over your text corpus, and
- an optional **Neo4j Knowledge Graph** queried via LLM-generated Cypher,

then synthesising a grounded, cited answer with an OpenAI (or OpenAI-compatible) model.

> **Note on data & secrets.** This repository contains only *code and configuration*.
> The cultural-heritage dataset and all API keys/passwords are **not** included — you
> supply your own. See [Provide your dataset](#3-provide-your-dataset) and
> [Configure secrets](#4-configure-secrets-api-keys--passwords).

---

## Architecture

```
┌────────────┐        ┌─────────────────────────────┐        ┌────────────────┐
│  Open WebUI │──API──▶│  Pipelines (RAG + KG)        │──API──▶│  OpenAI        │
│  (frontend) │        │  KGSearchCultureBot_v2.py    │        │  (LLM + embed) │
└────────────┘        │  FAISS + BM25 + Cypher       │        └────────────────┘
      ▲                └──────────────┬──────────────┘
      │                               │ (kg / hybrid modes only)
   browser                           ▼
                              ┌────────────────┐
                              │  Neo4j graph   │  (optional)
                              └────────────────┘
```

Two containers run from `compose.yml`:

| Service | Image | Port (localhost) | Purpose |
|---------|-------|------------------|---------|
| `open-webui` | `ghcr.io/open-webui/open-webui:main` | `12012` | Chat UI + user auth |
| `pipelines`  | built from `sources/Dockerfile.pipelines` | `12011` | RAG/KG backend |

Optional services (commented out in `compose.yml`): a local **Ollama** backend and an
internal **auto-redeploy** webhook.

---

## Repository layout

```
.
├── compose.yml                     # Docker Compose stack (secrets via ${ENV})
├── .env.example                    # Template for your secrets — copy to .env
├── .gitignore                      # Keeps data & secrets out of the repo
├── data/                           # Open WebUI runtime data (gitignored, auto-created)
├── dataset/                        # YOUR dataset goes here (gitignored) — see its README
│   ├── README.md                   # Data formats + filenames the pipeline expects
│   ├── *.example.*                 # Small format samples (safe to publish)
│   └── kg_jsons/                   # Place taxonomy + filter vocab for the KG prompt
└── sources/
    ├── build.sh                    # Helper to build the pipelines image
    ├── Dockerfile.pipelines        # Builds the pipelines image
    └── pipelines/
        ├── KGSearchCultureBot_v2.py    # Main pipeline (KG + hybrid RAG)
        ├── SearchCultureBot.py         # Simpler RAG-only pipeline
        └── sc_kg_nl2cypher.py          # NL→Cypher helper module (deployed under kg_jsons/)
```

---

## Prerequisites

- A 64-bit Linux/macOS/Windows host with **Docker Engine 24+** and the **Docker
  Compose v2 plugin**.
- An **OpenAI API key** (or an OpenAI-compatible endpoint).
- *(Optional)* a **Neo4j** instance if you want the Knowledge-Graph / hybrid modes.
- Enough disk/RAM for your corpus and the FAISS index. A GPU is optional (only used by
  local HuggingFace embedding models; the default `text-embedding-3-small` uses the
  OpenAI API and needs no GPU).

### Install Docker & Docker Compose

**Ubuntu / Debian** (installs Docker Engine + the Compose v2 plugin):

```bash
# 1. Install using Docker's official convenience script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 2. Run docker without sudo (log out/in afterwards)
sudo usermod -aG docker "$USER"

# 3. Verify
docker --version
docker compose version
```

**macOS / Windows:** install [Docker Desktop](https://www.docker.com/products/docker-desktop/),
which bundles the Compose v2 plugin. Then verify with `docker compose version`.

The pipelines image installs CUDA-enabled PyTorch. If you have an NVIDIA GPU and want
to use it, also install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Otherwise the stack runs fine on CPU with the default OpenAI embeddings.

---

## Setup

### 1. Clone

```bash
git clone <your-fork-url> searchculturebot
cd searchculturebot
```

### 2. (Optional) build the pipelines image

`compose.yml` builds it automatically on `up`. To build it explicitly:

```bash
cd sources
./build.sh            # or: docker build -t searchculturebot-pipelines:latest -f Dockerfile.pipelines .
cd ..
```

### 3. Provide your dataset and enable the mounts

This repo ships **without** any dataset. You must add your own files to `dataset/`,
using the exact filenames and formats described in
**[`dataset/README.md`](dataset/README.md)**. In short:

- `dataset/puretext_chunks.jsonl` — `{"id": "<uri>", "text": "..."}` per line (KG lookup)
- `dataset/puretext_chunks.md` — one artifact per `##` section (RAG index)
- `dataset/kg_jsons/searchculture_places_taxonomy.json` — place taxonomy for the KG prompt
- `dataset/kg_jsons/searchculture_filters_from_page.json` — filter vocabulary for the KG prompt

Compare against the committed `*.example.*` files for the precise structure. These data
files are gitignored, so they will **never** be pushed to your public repo.

Then **uncomment the `volumes:` block** in the `pipelines` service of `compose.yml` so
those files are mounted into the container. It is commented out by default precisely
because no dataset is shipped — Docker would otherwise fail to bind-mount missing paths.

> **RAG-only?** If you have no Neo4j graph, you only need `puretext_chunks.md`. Set
> `QUERY_MODE` to `rag` from the Open WebUI admin panel (Admin Panel → Pipelines).

### 4. Configure secrets (API keys & passwords)

All secrets live in a local `.env` file that is **never committed**:

```bash
cp .env.example .env
# then edit .env
```

Fill in at least these values:

| Variable | Required | What it is |
|----------|----------|-----------|
| `OPENAI_API_KEY` | ✅ | Your OpenAI key — used for both generation and embeddings |
| `WEBUI_SECRET_KEY` | ✅ | Random secret signing Open WebUI sessions — `openssl rand -hex 32` |
| `PIPELINES_API_KEY` | ✅ | Random token Open WebUI uses to call the pipelines server — `openssl rand -hex 32` |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | Only for `kg`/`hybrid` modes | Your Neo4j connection |
| `OPENAI_MODEL`, `OPENAI_BASE_URL` | Optional | Model name / custom endpoint |
| `OLLAMA_BASE_URL`, `DEPLOY_AUTH_TOKEN` | Optional | Only for the optional services |

Generate the two random secrets like this:

```bash
openssl rand -hex 32   # use for WEBUI_SECRET_KEY
openssl rand -hex 32   # use for PIPELINES_API_KEY
```

> **How secrets reach the code:** `compose.yml` reads `.env` and injects the values as
> container environment variables. The pipeline code (`KGSearchCultureBot_v2.py` etc.)
> reads them via `os.getenv(...)`, so **no key is ever written in source**. No
> `valves.json` is baked into the image either — behavioural valves (models, retrieval
> sizes, query mode) fall back to the defaults coded in each pipeline's `Valves` class,
> and you can tune them live from the Open WebUI **Admin Panel → Pipelines**.

### 5. Start the stack

```bash
docker compose up -d
docker compose logs -f pipelines   # watch the first-run index build
```

Open WebUI is now at **http://localhost:12012**. The first pipeline run builds the FAISS
+ BM25 index from your corpus (this can take a few minutes for a large dataset) and
caches it for subsequent restarts.

### 6. Connect Open WebUI to the pipeline

1. Open http://localhost:12012 and create the first account (it becomes the admin).
2. Go to **Admin Panel → Settings → Connections**.
3. Add an **OpenAI API** connection:
   - **Base URL:** `http://pipelines:9099`
   - **API Key:** the value of `PIPELINES_API_KEY` from your `.env`
4. The `KGSearchCultureBot` pipeline now appears as a selectable model in new chats.

---

## Query modes

Set `QUERY_MODE` in the valves (or send `[mode:hybrid|kg|rag]` in a system message):

- **`hybrid`** (default) — Knowledge Graph **and** RAG merged into one grounded answer.
- **`kg`** — LLM generates Cypher → Neo4j → artifact lookup → answer. Needs Neo4j.
- **`rag`** — FAISS + BM25 hybrid search only. No Neo4j required.

Embedding backends (valve `EMBEDDING_MODEL`): `text-embedding-3-small` (default, OpenAI),
`text-embedding-3-large` (OpenAI), `sentence-transformers/all-MiniLM-L6-v2` (local, CPU),
`Qwen/Qwen3-Embedding-0.6B` (local, GPU).

---

## Security notes

- **Never commit `.env` or your real `dataset/` data** — both are gitignored. Verify
  with `git status` before your first push.
- If any key in this repo's history was ever real, **rotate it** (issue a new OpenAI
  key, Neo4j password, etc.) before publishing. The previously hard-coded values have
  been removed from the code, but a public git history can still expose old commits.
- Container ports are bound to `127.0.0.1` only. For public access, put a reverse proxy
  (e.g. Caddy/Nginx) with TLS in front, and set `WEBUI_SESSION_COOKIE_SECURE=true`.
- Keep `DEFAULT_USER_ROLE=pending` so new sign-ups require admin approval.

---

## Troubleshooting

- **`pipelines` errors that data files are missing** — check the filenames in
  `dataset/` against `dataset/README.md`; they are bind-mounted by exact name.
- **Compose refuses to start with `set WEBUI_SECRET_KEY in .env`** — a required value is
  missing from `.env`. Fill it in.
- **KG/hybrid mode returns nothing** — verify `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD`
  and that your graph uses the same artifact ids as `puretext_chunks.jsonl`.
- **Rebuild after changing pipeline code or `Dockerfile.pipelines`:**
  `docker compose build pipelines && docker compose up -d`.
- **Force a fresh index** — delete the cache volume/dir inside the container, or bump a
  retrieval valve (the cache key includes them) and restart.
