"""Command-line interface.

Commands map 1:1 onto the report:
    gen-samples   generate synthetic insurance PDFs (corpus)
    ingest        load -> chunk -> embed -> store in ChromaDB   (report §5)
    query         RAG answer with citations                     (report §8)
    baseline      LLM-only answer, no retrieval                 (report §3)
    prompted      prompt-engineered LLM answer, no retrieval    (report §4)
    compare       baseline vs prompted vs RAG for one question
    evaluate      score all three modes on the canonical queries (report §8.4)
    sweep         run the five fine-tuning configs C1-C5        (report §9)
    info          environment / index status
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import load_config
from .llm import LLMUnavailable, OllamaLLM

console = Console()


def _cfg(config_path):
    return load_config(config_path)


def _fail(msg: str) -> None:
    console.print(f"[bold red]Error:[/] {msg}")
    raise SystemExit(1)


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config.yaml")
@click.pass_context
def cli(ctx, config_path):
    """Insurance RAG Q&A Bot."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


# --------------------------------------------------------------------------
@cli.command()
@click.pass_context
def info(ctx):
    """Show configuration, corpus, index, and LLM status."""
    cfg = _cfg(ctx.obj["config_path"])
    from .ingest import list_pdfs

    pdfs = list_pdfs(cfg.docs_path)
    table = Table(title="Insurance RAG - status", show_header=False)
    table.add_row("Docs dir", str(cfg.docs_path))
    table.add_row("PDFs found", str(len(pdfs)))
    table.add_row("Embedding model", cfg.embedding.model_name)
    table.add_row("Chunk size / overlap", f"{cfg.chunking.chunk_size} / {cfg.chunking.chunk_overlap}")
    table.add_row("Retriever k", str(cfg.retriever.k))
    table.add_row("LLM", f"{cfg.llm.provider}:{cfg.llm.model} @ {cfg.llm.host}")

    # collection size (best-effort)
    try:
        from .pipeline import build_vectorstore
        store = build_vectorstore(cfg)
        table.add_row("Vectors stored", str(store.count()))
    except Exception as e:
        table.add_row("Vectors stored", f"[yellow]unavailable ({e})[/]")

    # ollama status
    try:
        models = OllamaLLM(cfg.llm.model, cfg.llm.host).available_models()
        table.add_row("Ollama models", ", ".join(models) or "none")
    except LLMUnavailable as e:
        table.add_row("Ollama", f"[yellow]{e}[/]")
    console.print(table)


# --------------------------------------------------------------------------
@cli.command("gen-samples")
@click.option("--overwrite", is_flag=True, help="Regenerate even if files exist")
@click.pass_context
def gen_samples(ctx, overwrite):
    """Generate synthetic insurance PDFs into the synthetic docs dir."""
    cfg = _cfg(ctx.obj["config_path"])
    from .sample_data import generate_samples

    paths = generate_samples(cfg.synthetic_path, overwrite=overwrite)
    console.print(f"[green]Wrote {len(paths)} sample PDFs[/] to {cfg.synthetic_path}")
    for p in paths:
        console.print(f"  - {Path(p).name}")


# --------------------------------------------------------------------------
@cli.command()
@click.option("--reset", is_flag=True, help="Clear the collection before ingesting")
@click.pass_context
def ingest(ctx, reset):
    """Ingest all PDFs under the docs dir into the vector database."""
    cfg = _cfg(ctx.obj["config_path"])
    from .pipeline import ingest as run_ingest

    with console.status("Loading, chunking, embedding (first run downloads gte-large)..."):
        rep = run_ingest(cfg, reset=reset)
    table = Table(title="Ingestion complete", show_header=False)
    table.add_row("Pages loaded", str(rep.pages))
    table.add_row("Chunks created", str(rep.chunks))
    table.add_row("Chunks stored", str(rep.stored))
    table.add_row("Collection size", str(rep.collection_size))
    console.print(table)
    if rep.chunks == 0:
        console.print(
            "[yellow]No documents found.[/] Add PDFs to the docs dir or run "
            "`insurance-rag gen-samples` first."
        )


# --------------------------------------------------------------------------
def _print_answer(result):
    console.print(Panel(result.answer, title="Answer", border_style="green"))
    if result.sources:
        console.print("[bold]Sources:[/] " + "; ".join(result.sources))


@cli.command()
@click.argument("question")
@click.option("--k", type=int, default=None, help="Number of chunks to retrieve")
@click.option("--show-context", is_flag=True, help="Print retrieved context")
@click.pass_context
def query(ctx, question, k, show_context):
    """Answer a QUESTION with the grounded RAG pipeline (report §8)."""
    cfg = _cfg(ctx.obj["config_path"])
    from .pipeline import RAGPipeline

    try:
        pipe = RAGPipeline(cfg)
        pipe.llm.ensure_ready()
        if show_context:
            contexts = pipe.retriever.retrieve(question, k=k)
            from .retriever import format_context
            console.print(Panel(format_context(contexts), title="Retrieved context", border_style="blue"))
        result = pipe.answer(question, k=k)
    except LLMUnavailable as e:
        _fail(str(e))
    _print_answer(result)


@cli.command()
@click.argument("question")
@click.pass_context
def baseline(ctx, question):
    """Answer with the raw LLM, no retrieval (report §3)."""
    cfg = _cfg(ctx.obj["config_path"])
    from .pipeline import RAGPipeline

    try:
        ans = RAGPipeline(cfg).baseline(question)
    except LLMUnavailable as e:
        _fail(str(e))
    console.print(Panel(ans, title="Baseline (no retrieval)", border_style="yellow"))


@cli.command()
@click.argument("question")
@click.pass_context
def prompted(ctx, question):
    """Answer with prompt engineering, no retrieval (report §4)."""
    cfg = _cfg(ctx.obj["config_path"])
    from .pipeline import RAGPipeline

    try:
        ans = RAGPipeline(cfg).prompted(question)
    except LLMUnavailable as e:
        _fail(str(e))
    console.print(Panel(ans, title="Prompt-engineered (no retrieval)", border_style="yellow"))


@cli.command()
@click.argument("question")
@click.pass_context
def compare(ctx, question):
    """Show baseline vs prompted vs RAG answers for one QUESTION."""
    cfg = _cfg(ctx.obj["config_path"])
    from .pipeline import RAGPipeline

    try:
        pipe = RAGPipeline(cfg)
        pipe.llm.ensure_ready()
        console.print(Panel(pipe.baseline(question), title="1. Baseline (no retrieval)", border_style="yellow"))
        console.print(Panel(pipe.prompted(question), title="2. Prompt-engineered (no retrieval)", border_style="yellow"))
        rag = pipe.answer(question)
    except LLMUnavailable as e:
        _fail(str(e))
    console.print(Panel(rag.answer, title="3. RAG (grounded)", border_style="green"))
    if rag.sources:
        console.print("[bold]Sources:[/] " + "; ".join(rag.sources))


# --------------------------------------------------------------------------
@cli.command()
@click.option("--no-judge", is_flag=True, help="Skip LLM-as-a-Judge scoring")
@click.option("--json", "json_out", type=click.Path(), default=None, help="Write results to JSON")
@click.pass_context
def evaluate(ctx, no_judge, json_out):
    """Compare baseline/prompted/RAG across the canonical queries (report §8.4)."""
    cfg = _cfg(ctx.obj["config_path"])
    from .experiments import evaluate_modes

    try:
        with console.status("Running evaluation (this issues several LLM calls)..."):
            results = evaluate_modes(cfg, use_judge=not no_judge)
    except LLMUnavailable as e:
        _fail(str(e))

    table = Table(title="Evaluation: baseline vs prompted vs RAG")
    for col in ["Query", "Mode", "Grounded", "Relevance", "ROUGE-L", "BLEU", "J:Ground", "J:Rel", "J:Coh"]:
        table.add_column(col)
    for r in results:
        table.add_row(
            r.question[:32] + ("..." if len(r.question) > 32 else ""),
            r.mode,
            f"{r.groundedness:.3f}",
            f"{r.relevance:.3f}",
            "-" if r.rouge_l is None else f"{r.rouge_l:.3f}",
            "-" if r.bleu is None else f"{r.bleu:.1f}",
            str(r.judge_groundedness or "-"),
            str(r.judge_relevance or "-"),
            str(r.judge_coherence or "-"),
        )
    console.print(table)
    if json_out:
        Path(json_out).write_text(json.dumps([r.as_row() for r in results], indent=2), encoding="utf-8")
        console.print(f"[green]Wrote[/] {json_out}")


# --------------------------------------------------------------------------
@cli.command()
@click.option("--json", "json_out", type=click.Path(), default=None, help="Write results to JSON")
@click.pass_context
def sweep(ctx, json_out):
    """Run the five RAG fine-tuning configurations C1-C5 (report §9)."""
    cfg = _cfg(ctx.obj["config_path"])
    from .experiments import run_sweep

    try:
        with console.status("Running 5 configs x queries..."):
            res = run_sweep(cfg)
    except LLMUnavailable as e:
        _fail(str(e))

    table = Table(title="RAG fine-tuning sweep (avg over queries)")
    for col in ["Config", "k", "temp", "max_tokens", "Avg Grounded", "Avg Relevance"]:
        table.add_column(col)
    for s in res.summary:
        table.add_row(
            s.config, str(s.k), f"{s.temperature:g}", str(s.max_tokens),
            f"{s.avg_groundedness:.3f}", f"{s.avg_relevance:.3f}",
        )
    console.print(table)
    if json_out:
        payload = {
            "summary": [s.__dict__ for s in res.summary],
            "cells": [c.__dict__ for c in res.cells],
        }
        Path(json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]Wrote[/] {json_out}")


if __name__ == "__main__":
    cli()
