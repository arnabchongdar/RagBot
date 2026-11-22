"""Fast smoke tests that do not require the LLM or the embedding model download.

Run with:  pytest -q
"""

from __future__ import annotations

from insurance_rag import prompts
from insurance_rag.chunk import chunk_pages
from insurance_rag.config import load_config
from insurance_rag.ingest import LoadedPage
from insurance_rag.retriever import format_context, source_label
from insurance_rag.sample_data import generate_samples
from insurance_rag.vectorstore import Retrieved


def test_config_loads_report_values():
    cfg = load_config()
    assert cfg.chunking.chunk_size == 512
    assert cfg.chunking.chunk_overlap == 20
    assert cfg.retriever.k == 3
    assert cfg.embedding.model_name == "thenlper/gte-large"
    assert len(cfg.sweep_configs) == 5
    assert [c.name for c in cfg.sweep_configs][0] == "C1_Baseline"
    assert len(cfg.queries) == 3
    assert all(q.gold for q in cfg.queries)  # gold answers enable ROUGE/BLEU


def test_chunking_splits_and_keeps_metadata():
    text = ". ".join(f"Clause number {i} about coverage and exclusions" for i in range(80))
    pages = [LoadedPage(text=text, metadata={"source": "policy.pdf", "page": 0})]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert len({c.id for c in chunks}) == len(chunks)  # unique ids
    assert all(c.metadata["source"] == "policy.pdf" for c in chunks)
    assert all(len(c.text) <= 260 for c in chunks)  # size + overlap slack


def test_prompt_template_composition():
    assert prompts.SYSTEM_GROUNDED
    assert prompts.SYSTEM_BASELINE == ""
    msg = prompts.build_grounded_user_message("[S1] (a.pdf, p.1)\nfoo", "Is X covered?")
    assert "foo" in msg and "Is X covered?" in msg


def test_format_context_numbers_and_labels_sources():
    r = Retrieved(text="water damage is covered", metadata={"source": "home.pdf", "page": 1}, score=0.1)
    block = format_context([r])
    assert "[S1]" in block
    assert source_label(r.metadata) == "home.pdf, p.2"  # page is 0-indexed internally
    assert format_context([]) == "(no relevant context retrieved)"


def test_generate_samples(tmp_path):
    paths = generate_samples(tmp_path, overwrite=True)
    assert len(paths) == 4
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
