"""Experiments (report §3, §4, §8, §9).

Two experiment drivers:
  * :func:`evaluate_modes` - compare baseline vs prompt-engineered vs RAG on the
    canonical queries, scoring each answer (report §3/§4/§8).
  * :func:`run_sweep` - run the five RAG fine-tuning configurations (C1-C5)
    across the queries and capture groundedness/relevance (report §9).

For a fair contrast, baseline/prompted answers are scored for groundedness
against the same retrieved context RAG would use, even though those modes do
not see it. This directly demonstrates the report's thesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import evaluate as ev
from .config import Config
from .embed import Embedder
from .llm import OllamaLLM
from .pipeline import RAGPipeline
from .retriever import format_context


def evaluate_modes(
    cfg: Config,
    *,
    pipeline: RAGPipeline | None = None,
    modes: tuple[str, ...] = ("baseline", "prompted", "rag"),
    use_judge: bool = True,
    progress=None,
) -> list[ev.EvalResult]:
    pipe = pipeline or RAGPipeline(cfg)
    embedder = pipe.embedder
    judge = OllamaLLM(cfg.evaluation.judge_model, cfg.llm.host) if use_judge else None

    results: list[ev.EvalResult] = []
    for q in cfg.queries:
        contexts = pipe.retriever.retrieve(q.question)
        context_text = format_context(contexts)
        for mode in modes:
            if progress:
                progress(f"[{mode}] {q.id}")
            if mode == "baseline":
                answer = pipe.baseline(q.question)
            elif mode == "prompted":
                answer = pipe.prompted(q.question)
            else:
                answer = pipe.answer(q.question).answer

            judged = (
                ev.llm_judge(q.question, answer, context_text, judge)
                if judge
                else {"groundedness": 0, "relevance": 0, "coherence": 0}
            )
            results.append(
                ev.EvalResult(
                    question=q.question,
                    mode=mode,
                    answer=answer,
                    groundedness=ev.groundedness_score(answer, contexts, embedder),
                    relevance=ev.relevance_score(answer, q.question, embedder),
                    rouge_l=ev.rouge_l(answer, q.gold or ""),
                    bleu=ev.bleu(answer, q.gold or ""),
                    judge_groundedness=judged["groundedness"],
                    judge_relevance=judged["relevance"],
                    judge_coherence=judged["coherence"],
                )
            )
    return results


# --------------------------------------------------------------------------
# Fine-tuning sweep (report §9)
# --------------------------------------------------------------------------
@dataclass
class SweepCell:
    config: str
    query_id: str
    k: int
    temperature: float
    max_tokens: int
    groundedness: float
    relevance: float
    answer: str


@dataclass
class SweepSummary:
    config: str
    k: int
    temperature: float
    max_tokens: int
    avg_groundedness: float
    avg_relevance: float


@dataclass
class SweepResult:
    cells: list[SweepCell] = field(default_factory=list)
    summary: list[SweepSummary] = field(default_factory=list)


def run_sweep(
    cfg: Config,
    *,
    pipeline: RAGPipeline | None = None,
    progress=None,
) -> SweepResult:
    pipe = pipeline or RAGPipeline(cfg)
    embedder = pipe.embedder
    cells: list[SweepCell] = []

    for sc in cfg.sweep_configs:
        for q in cfg.queries:
            if progress:
                progress(f"[{sc.name}] {q.id} (k={sc.k}, T={sc.temperature})")
            result = pipe.answer(
                q.question,
                k=sc.k,
                temperature=sc.temperature,
                top_p=sc.top_p,
                top_k=sc.top_k,
                max_tokens=sc.max_tokens,
            )
            g = ev.groundedness_score(result.answer, result.contexts, embedder)
            r = ev.relevance_score(result.answer, q.question, embedder)
            cells.append(
                SweepCell(
                    config=sc.name,
                    query_id=q.id,
                    k=sc.k,
                    temperature=sc.temperature,
                    max_tokens=sc.max_tokens,
                    groundedness=g,
                    relevance=r,
                    answer=result.answer,
                )
            )

    # aggregate per config
    summary: list[SweepSummary] = []
    for sc in cfg.sweep_configs:
        rows = [c for c in cells if c.config == sc.name]
        n = len(rows) or 1
        summary.append(
            SweepSummary(
                config=sc.name,
                k=sc.k,
                temperature=sc.temperature,
                max_tokens=sc.max_tokens,
                avg_groundedness=round(sum(c.groundedness for c in rows) / n, 4),
                avg_relevance=round(sum(c.relevance for c in rows) / n, 4),
            )
        )
    return SweepResult(cells=cells, summary=summary)
