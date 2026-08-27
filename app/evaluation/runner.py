"""Eval runner: run configurations over the question set and compare them.

The point is not one number. It is the COMPARISON -- running the same
questions through vector-only, lexical-only, hybrid, and hybrid+rerank, and
reading off what each stage actually bought. That is the difference between
"we added a reranker" and "the reranker moved MRR from 0.61 to 0.78".
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from ..answer import Answer, AnswerGenerator
from ..config import Settings, get_settings
from ..llm import LLM
from ..retrieval import Retriever
from .dataset import EvalQuestion
from .metrics import (
    Judgement, citation_validity, hit_at_k, is_grounded, judge_answer, nanmean,
    precision_at_k, recall_at_k, reciprocal_rank, refusal_correct,
    required_content_coverage,
)


@dataclass(slots=True)
class Config:
    """One retrieval configuration to evaluate."""

    name: str
    use_vector: bool = True
    use_lexical: bool = True
    use_rewrite: bool = False
    use_rerank: bool = False
    top_k: int | None = None
    top_n: int | None = None
    rrf_k: int = 60

    @property
    def weights(self) -> list[float] | None:
        # Zero-weighting a retriever isolates the other one while keeping the
        # rest of the pipeline byte-identical -- so a difference in the numbers
        # is attributable to the retriever and nothing else.
        if self.use_vector and self.use_lexical:
            return None
        return [1.0 if self.use_lexical else 0.0, 1.0 if self.use_vector else 0.0]


DEFAULT_CONFIGS = [
    Config("lexical-only", use_vector=False),
    Config("vector-only", use_lexical=False),
    Config("hybrid"),
    Config("hybrid+rewrite", use_rewrite=True),
    Config("hybrid+rerank", use_rerank=True),
    Config("full", use_rewrite=True, use_rerank=True),
]


@dataclass(slots=True)
class QuestionResult:
    question_id: str
    question: str
    answerable: bool
    retrieved: list[str]
    final: list[str]
    relevant: list[str]
    latency_ms: float
    answer_text: str = ""
    refused: bool = False
    invalid_citations: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    judgement: Judgement | None = None


@dataclass(slots=True)
class Report:
    config: str
    results: list[QuestionResult]
    aggregates: dict[str, float]
    usage: str = ""
    seconds: float = 0.0

    def failures(self, limit: int = 5) -> list[QuestionResult]:
        """Questions whose relevant chunk never made the final cut.

        The most useful output of the whole harness: week 7 of the plan is
        "read failure cases, fix retrieval/prompts", and this is that list.
        """
        missed = [r for r in self.results
                  if r.answerable and not (set(r.final) & set(r.relevant))]
        return sorted(missed, key=lambda r: r.metrics.get("mrr", 0.0))[:limit]


class Harness:
    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: AnswerGenerator | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = LLM(self.settings)
        self.retriever = retriever or Retriever(llm=self.llm, settings=self.settings)
        self.generator = generator or AnswerGenerator(llm=self.llm, settings=self.settings)

    def run(
        self,
        questions: Sequence[EvalQuestion],
        config: Config,
        *,
        generate: bool = False,
        judge: bool = False,
        k_values: Sequence[int] = (1, 3, 5, 10),
    ) -> Report:
        started = time.time()
        results: list[QuestionResult] = []

        for q in questions:
            t0 = time.time()
            retrieval = self.retriever.retrieve(
                q.question,
                top_k=config.top_k or self.settings.retrieval_top_k,
                top_n=config.top_n or self.settings.rerank_top_n,
                use_rewrite=config.use_rewrite,
                use_rerank=config.use_rerank,
                rrf_k=config.rrf_k,
                weights=config.weights,
            )
            fused = retrieval.stages.get("fused", [])
            final = retrieval.stages.get("final", [])

            m: dict[str, float] = {"mrr": reciprocal_rank(fused, q.relevant_chunk_ids)}
            for k in k_values:
                m[f"recall@{k}"] = recall_at_k(fused, q.relevant_chunk_ids, k)
                m[f"hit@{k}"] = hit_at_k(fused, q.relevant_chunk_ids, k)
            m[f"precision@{len(final) or 1}"] = precision_at_k(
                final, q.relevant_chunk_ids, len(final) or 1)

            answer: Answer | None = None
            judgement: Judgement | None = None
            if generate:
                answer = self.generator.answer(q.question, retrieval)
                m["citation_validity"] = citation_validity(answer)
                m["grounded"] = 1.0 if is_grounded(answer) else 0.0
                m["refusal_correct"] = 1.0 if refusal_correct(q, answer) else 0.0
                m["required_content"] = required_content_coverage(q, answer)
                m["uncited_claims"] = float(answer.uncited_claims)
                if judge:
                    judgement = judge_answer(q, answer, self.llm, self.settings)
                    m["faithfulness"] = judgement.faithfulness
                    m["relevance"] = judgement.relevance

            results.append(QuestionResult(
                question_id=q.id, question=q.question, answerable=q.answerable,
                retrieved=fused, final=final, relevant=q.relevant_chunk_ids,
                latency_ms=(time.time() - t0) * 1000,
                answer_text=answer.text if answer else "",
                refused=answer.refused if answer else False,
                invalid_citations=answer.invalid_citations if answer else [],
                metrics=m, judgement=judgement,
            ))

        keys = sorted({k for r in results for k in r.metrics})
        aggregates = {k: nanmean([r.metrics.get(k, float("nan")) for r in results]) for k in keys}
        aggregates["latency_ms_p50"] = sorted(r.latency_ms for r in results)[len(results) // 2]

        return Report(config=config.name, results=results, aggregates=aggregates,
                      usage=self.llm.usage.report(), seconds=time.time() - started)


def comparison_table(reports: Sequence[Report], columns: Sequence[str]) -> str:
    """Configurations as rows, metrics as columns."""
    width = max(len(r.config) for r in reports) + 2
    head = f"{'config':<{width}}" + "".join(f"{c:>18}" for c in columns)
    lines = [head, "-" * len(head)]
    for report in reports:
        row = f"{report.config:<{width}}"
        for column in columns:
            value = report.aggregates.get(column, float("nan"))
            row += f"{'—':>18}" if value != value else f"{value:>18.3f}"
        lines.append(row)
    return "\n".join(lines)


def report_dict(report: Report) -> dict[str, Any]:
    return {
        "config": report.config,
        "seconds": round(report.seconds, 2),
        "usage": report.usage,
        "aggregates": {k: (None if v != v else round(v, 4))
                       for k, v in report.aggregates.items()},
        "results": [
            {**asdict(r), "judgement": asdict(r.judgement) if r.judgement else None}
            for r in report.results
        ],
    }
