"""
BC-Score Report Generator

Produces human-readable reports and machine-readable summaries from evaluation results.
"""

import json
from datetime import datetime
from typing import Optional

from scoring.bc_score import BCScore, aggregate_scores


def _bar(score: float, width: int = 20) -> str:
    """Create a visual bar for terminal output."""
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


def _grade(score: float) -> str:
    """Convert score to letter grade."""
    if score >= 0.9:
        return "A"
    elif score >= 0.8:
        return "A-"
    elif score >= 0.7:
        return "B+"
    elif score >= 0.6:
        return "B"
    elif score >= 0.5:
        return "C+"
    elif score >= 0.4:
        return "C"
    elif score >= 0.3:
        return "D"
    else:
        return "F"


def print_single_report(score: BCScore, model_name: str = "Unknown Model"):
    """Print a formatted BC-Score report for a single evaluation."""
    print()
    print("=" * 60)
    print(f"  ContinuityBench · BC-Score Report")
    print(f"  Model: {model_name}")
    print(f"  Date:  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print()

    composite = score.composite
    print(f"  COMPOSITE BC-SCORE:  {composite:.3f}  [{_grade(composite)}]")
    print(f"  {_bar(composite, 40)}")
    print()

    print("  Per-Dimension Scores:")
    print("  " + "-" * 56)

    for dim_name, dim_score in [
        ("Identity    ", score.identity),
        ("Goal        ", score.goal),
        ("Abstraction ", score.abstraction),
        ("Style       ", score.style),
    ]:
        conf_str = f"(conf: {dim_score.confidence:.2f})"
        print(
            f"  {dim_name} {_bar(dim_score.score)} "
            f"{dim_score.score:.3f} [{_grade(dim_score.score)}] {conf_str}"
        )

    print()
    print(f"  Weakest Dimension: {score.min_dimension.dimension.upper()}")
    print(f"  Mean Judge Agreement: {score.mean_confidence:.3f}")

    if score.auxiliary:
        print()
        print("  Auxiliary Metrics:")
        print("  " + "-" * 56)
        print(f"  NPC Fallback Rate:      {score.auxiliary.npc_fallback_rate:.1%}")
        print(f"  Goal Recovery Latency:  {score.auxiliary.goal_recovery_latency:.1f} turns")
        print(f"  Tone Variance:          {score.auxiliary.tone_variance:.4f}")
        print(f"  Self-Inconsistency:     {score.auxiliary.self_inconsistency_rate:.1%}")

    # Evidence summary
    print()
    print("  Key Evidence:")
    print("  " + "-" * 56)
    for dim in [score.identity, score.goal, score.abstraction, score.style]:
        if dim.evidence:
            print(f"  [{dim.dimension.upper()}]")
            for ev in dim.evidence[:3]:
                print(f"    · {ev}")
            print()

    print("=" * 60)
    print()


def print_aggregate_report(
    scores: list[BCScore],
    model_name: str = "Unknown Model",
    weights: Optional[dict] = None,
):
    """Print an aggregate report across multiple evaluations."""
    agg = aggregate_scores(scores, weights)

    print()
    print("=" * 60)
    print(f"  ContinuityBench · Aggregate Report")
    print(f"  Model: {model_name}")
    print(f"  Evaluations: {agg['n_evaluations']}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print()

    c = agg["composite"]
    print(f"  COMPOSITE BC-SCORE:  {c['mean']:.3f} ± {c['std']:.3f}  [{_grade(c['mean'])}]")
    print(f"  Range: [{c['min']:.3f}, {c['max']:.3f}]")
    print(f"  {_bar(c['mean'], 40)}")
    print()

    print("  Per-Dimension Breakdown:")
    print("  " + "-" * 56)
    print(f"  {'Dimension':<14} {'Mean':>6} {'± Std':>7} {'Min':>6} {'Max':>6} {'Grade':>6}")
    print("  " + "-" * 56)
    for dim_name in ("identity", "goal", "abstraction", "style"):
        d = agg["dimensions"][dim_name]
        print(
            f"  {dim_name.capitalize():<14} {d['mean']:>6.3f} {d['std']:>6.3f} "
            f"{d['min']:>6.3f} {d['max']:>6.3f} {_grade(d['mean']):>6}"
        )
    print()
    print("=" * 60)
    print()


def save_json_report(
    scores: list[BCScore],
    model_name: str,
    output_path: str,
    metadata: Optional[dict] = None,
):
    """Save evaluation results as JSON."""
    agg = aggregate_scores(scores)
    report = {
        "meta": {
            "benchmark": "ContinuityBench",
            "version": "0.1.0",
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "n_evaluations": len(scores),
            **(metadata or {}),
        },
        "aggregate": agg,
        "individual": [s.to_dict() for s in scores],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Report saved to {output_path}")


def generate_report(
    scores: list[BCScore],
    model_name: str = "Unknown Model",
    output_path: Optional[str] = None,
    weights: Optional[dict] = None,
):
    """
    Generate and display a BC-Score report.

    If output_path is provided, also saves a JSON report.
    """
    if len(scores) == 1:
        print_single_report(scores[0], model_name)
    else:
        print_aggregate_report(scores, model_name, weights)

    if output_path:
        save_json_report(scores, model_name, output_path)
