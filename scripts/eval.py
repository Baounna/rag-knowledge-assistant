#!/usr/bin/env python3
"""Run the evaluation harness.

    make eval                 # retrieval metrics, all configs, no API key
    make eval-full            # + generation, citations, refusal, LLM judge
    python3 scripts/eval.py --config hybrid --failures 10

Retrieval metrics need no API key. Answer metrics need one.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.evaluation import (  # noqa: E402
    DEFAULT_CONFIGS, Config, Harness, comparison_table, coverage, load_questions, report_dict,
)
from app.store import Store  # noqa: E402

RETRIEVAL_COLUMNS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "latency_ms_p50"]
ANSWER_COLUMNS = ["grounded", "citation_validity", "refusal_correct",
                  "required_content", "faithfulness", "relevance"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    ap.add_argument("--config", action="append", default=None,
                    help="config name to run; repeatable. default: all")
    ap.add_argument("--generate", action="store_true", help="also generate answers")
    ap.add_argument("--judge", action="store_true", help="also run the LLM judge")
    ap.add_argument("--failures", type=int, default=5, help="failure cases to print")
    ap.add_argument("--out", type=Path, default=Path("eval/report.json"))
    args = ap.parse_args()

    settings = get_settings()
    questions = load_questions(args.questions)
    store = Store(settings)
    try:
        store.init_schema()
        with store.conn() as c, c.cursor() as cur:
            cur.execute("SELECT chunk_id FROM chunks")
            indexed = {r["chunk_id"] for r in cur.fetchall()}
    except Exception as exc:  # noqa: BLE001
        print(f"database unavailable ({type(exc).__name__}) -- run `make db-up && make index`")
        return 1
    if not indexed:
        print("index is empty -- run `make ingest && make index`")
        return 1

    cov = coverage(questions, indexed)
    print("EVAL SET")
    print(f"  {cov['questions']} questions "
          f"({cov['answerable']} answerable, {cov['unanswerable']} unanswerable)")
    print(f"  labels cover {cov['corpus_covered']:.0%} of {cov['indexed_chunks']} indexed chunks")
    if cov["stale_labels"]:
        print(f"  !! {len(cov['stale_labels'])} labels point at chunks that no longer exist:")
        for cid in cov["stale_labels"][:5]:
            print(f"     {cid}")
        print("     the corpus was re-chunked -- relabel before trusting these numbers")
    if cov["questions"] < 50:
        print(f"  !! the subject asks for 50-100 questions; this set has {cov['questions']}")
    if cov["unanswerable"] == 0:
        print("  !! no unanswerable questions -- refusal is not being measured")

    # Degenerate-eval guard. If K reaches most of the corpus, every retriever
    # returns nearly everything and recall@K is ~1.0 by construction -- the
    # metric measures the corpus size, not the retriever. This is exactly the
    # "eval theater" failure: numbers that look excellent and mean nothing.
    if settings.retrieval_top_k >= cov["indexed_chunks"]:
        print(f"\n  !! RETRIEVAL_TOP_K={settings.retrieval_top_k} >= "
              f"{cov['indexed_chunks']} indexed chunks.")
        print("     Every retriever returns the whole corpus, so recall@K is 1.0 by")
        print("     construction and tells you nothing. Only MRR is meaningful here.")
        print("     Index a real corpus before trusting recall.")
    elif settings.retrieval_top_k > cov["indexed_chunks"] * 0.5:
        print(f"\n  !! K={settings.retrieval_top_k} covers over half the corpus "
              f"({cov['indexed_chunks']} chunks) -- recall is inflated.")

    if not settings.anthropic_api_key:
        print("\n  !! no ANTHROPIC_API_KEY: rewriting and reranking silently do nothing,")
        print("     so 'hybrid', 'hybrid+rewrite', 'hybrid+rerank' and 'full' are")
        print("     the same pipeline. Identical rows below are expected, not a result.")

    if (args.generate or args.judge) and not settings.anthropic_api_key:
        print("\nANTHROPIC_API_KEY is not set -- skipping generation and judging")
        args.generate = args.judge = False

    wanted = args.config or [c.name for c in DEFAULT_CONFIGS]
    by_name = {c.name: c for c in DEFAULT_CONFIGS}
    unknown = [n for n in wanted if n not in by_name]
    if unknown:
        print(f"unknown config(s): {unknown}. available: {sorted(by_name)}")
        return 1
    configs: list[Config] = [by_name[n] for n in wanted]

    harness = Harness(settings=settings)
    reports = []
    for config in configs:
        print(f"\nrunning {config.name} ...", end="", flush=True)
        report = harness.run(questions, config, generate=args.generate, judge=args.judge)
        print(f" {report.seconds:.1f}s")
        reports.append(report)

    print("\nRETRIEVAL")
    print(comparison_table(reports, RETRIEVAL_COLUMNS))
    if args.generate:
        print("\nANSWER")
        print(comparison_table(reports, ANSWER_COLUMNS))
        print(f"\nusage: {reports[-1].usage}")

    best = max(reports, key=lambda r: r.aggregates.get("mrr", 0.0))
    print(f"\nbest MRR: {best.config} ({best.aggregates.get('mrr', 0):.3f})")

    if args.failures:
        print(f"\nFAILURE CASES ({best.config}) -- relevant chunk missed entirely")
        misses = best.failures(args.failures)
        if not misses:
            print("  none")
        for r in misses:
            print(f"\n  {r.question_id}: {r.question}")
            print(f"    expected : {r.relevant}")
            print(f"    got      : {r.final[:3]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": settings.embedding_model,
        "coverage": cov,
        "reports": [report_dict(r) for r in reports],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
