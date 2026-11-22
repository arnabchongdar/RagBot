"""Streamlit web UI for the Insurance RAG Q&A bot.

Run it with:
    ./.venv/Scripts/streamlit.exe run app.py

Features:
  * drag-and-drop / upload PDF files into the corpus
  * a button to (re-)run ingestion: load -> chunk -> embed -> store
  * a chat box to ask grounded questions, with source citations
"""

from __future__ import annotations

import streamlit as st

from insurance_rag.chunk import chunk_pages
from insurance_rag.config import load_config
from insurance_rag.ingest import list_pdfs, load_documents
from insurance_rag.llm import LLMUnavailable
from insurance_rag.pipeline import RAGPipeline
from insurance_rag.retriever import format_context
from insurance_rag.sample_data import generate_samples

st.set_page_config(page_title="Insurance RAG Q&A", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------
# Cached resources (built once, reused across Streamlit reruns)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_config():
    return load_config()


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    # Cheap: constructs the embedder (model loads lazily on first embed),
    # the Chroma client, and the Ollama client. No heavy work until a query.
    return RAGPipeline(get_config())


cfg = get_config()
UPLOAD_DIR = cfg.docs_path / "uploads"


def index_count() -> int:
    try:
        return get_pipeline().store.count()
    except Exception:
        return -1


def ollama_status() -> tuple[bool, str]:
    try:
        models = get_pipeline().llm.available_models()
        base = cfg.llm.model.split(":")[0]
        ok = any(n == cfg.llm.model or n.split(":")[0] == base for n in models)
        return ok, (f"Ollama up · models: {', '.join(models)}" if ok
                    else f"Ollama up but '{cfg.llm.model}' not pulled")
    except LLMUnavailable as e:
        return False, str(e)


def run_ingest(reset: bool) -> tuple[int, int, int, int]:
    """Reuse the cached pipeline's Chroma client so queries see new data at once."""
    pipe = get_pipeline()
    if reset:
        pipe.store.reset()
    pages = load_documents(cfg.docs_path)
    chunks = chunk_pages(
        pages,
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
    )
    stored = pipe.store.add_chunks(chunks) if chunks else 0
    return len(pages), len(chunks), stored, pipe.store.count()


# --------------------------------------------------------------------------
# Sidebar: documents, ingest, settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Documents")

    uploaded = st.file_uploader(
        "Drop insurance PDFs here",
        type="pdf",
        accept_multiple_files=True,
        help="Files are saved into docs/uploads/ and included on the next ingest.",
    )

    reset = st.checkbox("Reset index before ingest", value=True,
                        help="Clear existing vectors first (recommended after changing files).")

    if st.button("⚙️  Save & (re-)run ingest", type="primary", use_container_width=True):
        saved = 0
        if uploaded:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            for uf in uploaded:
                (UPLOAD_DIR / uf.name).write_bytes(uf.getbuffer())
                saved += 1
        if saved:
            st.success(f"Saved {saved} file(s) to docs/uploads/")
        with st.spinner("Ingesting: load → chunk → embed → store…"):
            try:
                pages, chunks, stored, total = run_ingest(reset)
                if chunks == 0:
                    st.warning("No PDFs found. Upload files or generate samples below.")
                else:
                    st.success(f"Ingested {pages} pages → {chunks} chunks. Index size: {total}.")
            except Exception as e:  # surface any failure to the user
                st.error(f"Ingest failed: {e}")

    if st.button("✨  Generate synthetic samples", use_container_width=True):
        paths = generate_samples(cfg.synthetic_path, overwrite=False)
        st.info(f"{len(paths)} sample PDFs in docs/synthetic/. Now click ingest.")

    st.divider()
    st.subheader("Retrieval & generation")
    k = st.slider("Chunks to retrieve (k)", 1, 8, cfg.retriever.k)
    temperature = st.slider("Temperature", 0.0, 1.0, float(cfg.llm.temperature), 0.05)
    show_ctx = st.toggle("Show retrieved context", value=False)

    st.divider()
    n = index_count()
    pdfs = len(list_pdfs(cfg.docs_path))
    st.caption(f"**Corpus:** {pdfs} PDF(s)  ·  **Index:** {n if n >= 0 else '—'} chunks")
    ok, msg = ollama_status()
    (st.caption if ok else st.warning)(("🟢 " if ok else "🔴 ") + msg)


# --------------------------------------------------------------------------
# Main: chat
# --------------------------------------------------------------------------
st.title("🛡️ Insurance Policy & Claims Q&A")
st.caption("Answers are grounded in your uploaded insurance documents, with source citations.")

if index_count() == 0:
    st.info(
        "The index is empty. In the sidebar: **upload PDFs** (or **generate synthetic "
        "samples**), then click **Save & (re-)run ingest**. Then ask a question below."
    )

st.session_state.setdefault("messages", [])

# replay history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            st.caption("Sources: " + "; ".join(m["sources"]))

question = st.chat_input("Ask about coverage, claims, exclusions, deductibles…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        pipe = get_pipeline()
        if index_count() <= 0:
            st.warning("Please ingest some documents first (see the sidebar).")
        else:
            try:
                pipe.llm.ensure_ready()
                if show_ctx:
                    ctx = pipe.retriever.retrieve(question, k=k)
                    with st.expander("Retrieved context", expanded=False):
                        st.text(format_context(ctx))
                with st.spinner("Retrieving and generating a grounded answer…"):
                    result = pipe.answer(question, k=k, temperature=temperature)
                st.markdown(result.answer)
                if result.sources:
                    st.caption("Sources: " + "; ".join(result.sources))
                st.session_state.messages.append(
                    {"role": "assistant", "content": result.answer, "sources": result.sources}
                )
            except LLMUnavailable as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Something went wrong: {e}")
