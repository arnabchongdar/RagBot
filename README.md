# Insurance RAG Q&A Bot

A Retrieval-Augmented Generation (RAG) assistant that answers **insurance policy
and claims** questions grounded in approved documents, with source citations.
It implements the pipeline and experiments described in the project report
(*RAG-Based Insurance Policy and Claims Assistant*).

```
ingest → chunk → embed (MiniLM) → store (ChromaDB)
       → retrieve (top-k) → generate (Ollama / Mistral-7B) → grounded, cited answer
```

## What maps to the report

| Report section | Where it lives |
|---|---|
| §2.2 Local LLM (Mistral-7B-Instruct) | `llm.py` — via **Ollama** (see note below) |
| §3 Baseline LLM QA | `insurance-rag baseline` |
| §4 Prompt-engineered QA | `insurance-rag prompted` |
| §5.1/5.3 Load + chunk (512/20) | `ingest.py`, `chunk.py` |
| §5.4 Embeddings (Sentence-Transformers) | `embed.py` — default `all-MiniLM-L6-v2`, see **Performance** below |
| §5.5 Vector DB (ChromaDB) | `vectorstore.py` |
| §5.6 Retriever (similarity, k=3) | `retriever.py` |
| §5.7/5.8 Prompt template + response fn | `prompts.py`, `pipeline.py` |
| §8 RAG QA + evaluation | `insurance-rag query`, `insurance-rag evaluate` |
| §8.4 Groundedness / Relevance / ROUGE / BLEU / LLM-judge | `evaluate.py` |
| §9 Fine-tuning sweep (C1–C5) | `insurance-rag sweep` |

**Two deviations from the report**, both to keep a local CPU run reliable and fast —
both revertable in [`config.yaml`](config.yaml):

1. **Generator → Ollama.** Runs on **Ollama** with the `mistral` model instead of
   `llama-cpp-python`. Same Mistral-7B-Instruct family, fully local/offline
   (fulfils the report's "no external API" goal), but no C++ build or GGUF download.
2. **Embeddings → MiniLM.** Defaults to `sentence-transformers/all-MiniLM-L6-v2`
   instead of `thenlper/gte-large`. On a CPU-only machine gte-large embeds at
   ~2 s/chunk (≈130 s for the sample corpus); MiniLM is ~7× faster (≈17 s) with
   quality that is more than adequate here. See **Performance** below to swap back.

Everything else follows the report faithfully.

## Requirements

- **Python 3.10+** (verified working on 3.14).
- **[Ollama](https://ollama.com)** running locally with a Mistral model:
  ```bash
  ollama pull mistral
  ```
- First `ingest` downloads the embedding model from Hugging Face (default
  `all-MiniLM-L6-v2`, ~90 MB).

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # exposes the `insurance-rag` command
```

## Quick start (end to end)

```bash
insurance-rag gen-samples          # create synthetic insurance PDFs in docs/synthetic/
insurance-rag ingest --reset       # load → chunk → embed → store
insurance-rag query "Is water damage covered under my home insurance policy?"
```

Add your **own** PDFs to `docs/` (any subfolder) and re-run `ingest` to use them.

## Web UI

A [Streamlit](https://streamlit.io) app wraps the same pipeline so you can
**drag-and-drop PDFs**, click a button to **re-run ingestion**, and **ask
questions** in a chat box — no terminal needed.

```bash
pip install -e ".[ui]"     # or: pip install streamlit
streamlit run app.py
```

Then in the browser (defaults to http://localhost:8501):

1. **Upload** one or more PDFs in the sidebar (or click *Generate synthetic samples*).
2. Click **⚙️ Save & (re-)run ingest** — this loads → chunks → embeds → stores them.
3. Ask questions in the chat box; answers come back **grounded with source citations**.

Sliders let you change retrieval `k` and temperature on the fly, and a toggle
shows the exact retrieved context behind each answer. The sidebar reports corpus
size, index size, and live Ollama status.

## Commands

| Command | Description |
|---|---|
| `insurance-rag info` | Config, corpus size, index size, Ollama status |
| `insurance-rag gen-samples [--overwrite]` | Generate synthetic insurance PDFs |
| `insurance-rag ingest [--reset]` | Build/refresh the vector index |
| `insurance-rag query "..." [--k N] [--show-context]` | Grounded RAG answer + sources |
| `insurance-rag baseline "..."` | Raw LLM answer (no retrieval) |
| `insurance-rag prompted "..."` | Prompt-engineered answer (no retrieval) |
| `insurance-rag compare "..."` | All three side by side |
| `insurance-rag evaluate [--no-judge] [--json out.json]` | Score modes on canonical queries |
| `insurance-rag sweep [--json out.json]` | Run the five fine-tuning configs |

## Configuration

Everything tunable lives in [`config.yaml`](config.yaml): chunk size/overlap,
embedding model + device (`cpu`/`cuda`), retriever `k`, LLM model + generation
params, the five sweep configurations, and the canonical evaluation queries
(with optional gold answers that enable ROUGE/BLEU).

## Performance & the embedding model

Ingestion speed is dominated by the embedding model running on **CPU**. The
default is chosen for responsiveness on a laptop:

| `embedding.model_name` | Dim | Ingest speed (CPU) | Quality |
|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` *(default)* | 384 | fast (~0.3 s/chunk) | good |
| `thenlper/gte-small` | 384 | medium | better |
| `thenlper/gte-large` *(report §5.4)* | 1024 | slow (~2 s/chunk) | best |

A "hangs on a tiny PDF" symptom is almost always the model, not the file — the
`Loading weights` / HF-Hub lines are harmless; the wait is the silent CPU
embedding that follows.

To change models, edit `embedding.model_name` in [`config.yaml`](config.yaml),
then **re-ingest with reset** — the vector dimension changes, so old and new
vectors cannot coexist in the same index:

```bash
insurance-rag ingest --reset
```

In the web UI, tick **Reset index before ingest** and click **Save & (re-)run
ingest**. If you have an NVIDIA GPU, set `embedding.device: cuda` for a large
speedup with any model.

## Project layout

```
RAG/
├── config.yaml                 # all tunables
├── requirements.txt
├── pyproject.toml
├── app.py                      # Streamlit web UI (upload → ingest → ask)
├── docs/                       # your PDFs go here
│   ├── uploads/                # PDFs uploaded via the web UI
│   └── synthetic/              # generated sample PDFs
├── data/chroma/                # persisted ChromaDB (created on ingest)
└── src/insurance_rag/
    ├── config.py   ingest.py   chunk.py    embed.py
    ├── vectorstore.py          retriever.py
    ├── prompts.py  llm.py      pipeline.py
    ├── evaluate.py experiments.py  sample_data.py  cli.py
```

## Notes & limitations

- Answers are **informational** and grounded in the provided corpus; the system
  does not make formal coverage or claim decisions (enforced in the system prompt).
- LLM-as-a-Judge reuses the local model — acceptable for a prototype; production
  should use an independent judge or human review (report §8.4).
- Answer quality depends on the corpus. The synthetic docs are illustrative;
  swap in approved policy documents for real use.
