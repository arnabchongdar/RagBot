"""Output evaluation (report §8.4).

Metrics:
  * groundedness  - how much of the answer is supported by retrieved context
                    (embedding-similarity proxy) + optional LLM-as-a-Judge score
  * relevance     - how well the answer addresses the question
                    (embedding similarity) + optional LLM-as-a-Judge score
  * ROUGE-L / BLEU - reference-based, only when a gold answer is available
  * llm_judge     - scalable groundedness/relevance/coherence via a judge model

Note (report §8.4): using the same model family as judge is acceptable for a
prototype only; production should use an independent judge or human review.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from .embed import Embedder
from .llm import OllamaLLM
from .vectorstore import Retrieved

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _cosine(a: list[float], b: list[float]) -> float:
    import numpy as np

    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def groundedness_score(answer: str, contexts: list[Retrieved], embedder: Embedder) -> float:
    """Mean over answer sentences of the max cosine similarity to any context chunk.

    Ranges ~0-1; higher means the answer is more supported by retrieved text.
    """
    sents = _sentences(answer)
    if not sents or not contexts:
        return 0.0
    ctx_embs = embedder.embed_documents([c.text for c in contexts])
    sent_embs = embedder.embed_documents(sents)
    total = 0.0
    for se in sent_embs:
        total += max(_cosine(se, ce) for ce in ctx_embs)
    return round(total / len(sent_embs), 4)


def relevance_score(answer: str, question: str, embedder: Embedder) -> float:
    """Cosine similarity between the answer and the question embeddings."""
    if not answer.strip():
        return 0.0
    a, q = embedder.embed_documents([answer, question])
    return round(_cosine(a, q), 4)


def rouge_l(answer: str, gold: str) -> float | None:
    if not gold:
        return None
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return round(scorer.score(gold, answer)["rougeL"].fmeasure, 4)


def bleu(answer: str, gold: str) -> float | None:
    if not gold:
        return None
    import sacrebleu

    return round(sacrebleu.sentence_bleu(answer, [gold]).score, 2)


_JUDGE_PROMPT = (
    "You are a strict evaluator of an insurance assistant's answer.\n"
    "Rate the ANSWER on three axes, each an integer from 1 (poor) to 5 (excellent):\n"
    "- groundedness: is the answer supported by the CONTEXT (no invented facts)?\n"
    "- relevance: does it address the QUESTION?\n"
    "- coherence: is it clear and well structured?\n"
    "Respond with ONLY a JSON object like "
    '{{"groundedness": 4, "relevance": 5, "coherence": 4}}.\n\n'
    "QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nJSON:"
)


def llm_judge(
    question: str,
    answer: str,
    context: str,
    judge: OllamaLLM,
) -> dict[str, int]:
    """LLM-as-a-Judge scores (1-5) for groundedness, relevance, coherence."""
    prompt = _JUDGE_PROMPT.format(question=question, context=context or "(none)", answer=answer)
    raw = judge.generate(prompt, temperature=0.0, max_tokens=120)
    return _parse_judge(raw)


def _parse_judge(raw: str) -> dict[str, int]:
    default = {"groundedness": 0, "relevance": 0, "coherence": 0}
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        return default
    try:
        data = json.loads(match.group(0))
    except Exception:
        return default
    out = {}
    for key in default:
        try:
            out[key] = max(1, min(5, int(round(float(data.get(key, 0))))))
        except Exception:
            out[key] = 0
    return out


@dataclass
class EvalResult:
    question: str
    mode: str  # baseline | prompted | rag
    answer: str
    groundedness: float
    relevance: float
    rouge_l: float | None
    bleu: float | None
    judge_groundedness: int
    judge_relevance: int
    judge_coherence: int

    def as_row(self) -> dict:
        return asdict(self)
