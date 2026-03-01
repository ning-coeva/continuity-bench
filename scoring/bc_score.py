"""
BC-Score: Behavioral Continuity Score Calculator

Computes per-dimension scores and composite BC-Score from judge evaluations.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class DimensionScore:
    """Score for a single drift dimension."""
    dimension: str  # "identity", "goal", "abstraction", "style"
    score: float  # 0.0 - 1.0
    confidence: float  # inter-judge agreement
    evidence: list[str] = field(default_factory=list)  # key observations from judge

    def __post_init__(self):
        assert 0.0 <= self.score <= 1.0, f"Score must be 0-1, got {self.score}"
        assert self.dimension in ("identity", "goal", "abstraction", "style")


@dataclass
class AuxiliaryMetrics:
    """Auxiliary metrics beyond the four core dimensions."""
    npc_fallback_rate: float = 0.0      # template responses / total turns
    goal_recovery_latency: float = 0.0  # mean turns to resume after interruption
    tone_variance: float = 0.0          # std dev of style embeddings
    self_inconsistency_rate: float = 0.0  # contradictions / total claims


@dataclass
class BCScore:
    """Complete Behavioral Continuity Score for one evaluation run."""
    identity: DimensionScore
    goal: DimensionScore
    abstraction: DimensionScore
    style: DimensionScore
    auxiliary: Optional[AuxiliaryMetrics] = None

    # Default weights (can be overridden via config)
    DEFAULT_WEIGHTS = {
        "identity": 0.30,
        "goal": 0.25,
        "abstraction": 0.25,
        "style": 0.20,
    }

    @property
    def composite(self) -> float:
        """Compute weighted composite BC-Score."""
        return self.composite_with_weights(self.DEFAULT_WEIGHTS)

    def composite_with_weights(self, weights: dict[str, float]) -> float:
        """Compute composite score with custom weights."""
        assert abs(sum(weights.values()) - 1.0) < 0.01, "Weights must sum to 1.0"
        return (
            weights["identity"] * self.identity.score
            + weights["goal"] * self.goal.score
            + weights["abstraction"] * self.abstraction.score
            + weights["style"] * self.style.score
        )

    @property
    def min_dimension(self) -> DimensionScore:
        """Return the lowest-scoring dimension (weakest link)."""
        dims = [self.identity, self.goal, self.abstraction, self.style]
        return min(dims, key=lambda d: d.score)

    @property
    def mean_confidence(self) -> float:
        """Average inter-judge agreement across dimensions."""
        return np.mean([
            self.identity.confidence,
            self.goal.confidence,
            self.abstraction.confidence,
            self.style.confidence,
        ])

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        result = {
            "composite_score": round(self.composite, 4),
            "dimensions": {
                "identity": {
                    "score": round(self.identity.score, 4),
                    "confidence": round(self.identity.confidence, 4),
                    "evidence": self.identity.evidence,
                },
                "goal": {
                    "score": round(self.goal.score, 4),
                    "confidence": round(self.goal.confidence, 4),
                    "evidence": self.goal.evidence,
                },
                "abstraction": {
                    "score": round(self.abstraction.score, 4),
                    "confidence": round(self.abstraction.confidence, 4),
                    "evidence": self.abstraction.evidence,
                },
                "style": {
                    "score": round(self.style.score, 4),
                    "confidence": round(self.style.confidence, 4),
                    "evidence": self.style.evidence,
                },
            },
            "mean_confidence": round(self.mean_confidence, 4),
            "weakest_dimension": self.min_dimension.dimension,
        }
        if self.auxiliary:
            result["auxiliary"] = {
                "npc_fallback_rate": round(self.auxiliary.npc_fallback_rate, 4),
                "goal_recovery_latency": round(self.auxiliary.goal_recovery_latency, 2),
                "tone_variance": round(self.auxiliary.tone_variance, 4),
                "self_inconsistency_rate": round(self.auxiliary.self_inconsistency_rate, 4),
            }
        return result


def aggregate_scores(scores: list[BCScore], weights: Optional[dict] = None) -> dict:
    """
    Aggregate multiple BCScore results (e.g., across stressor variants).

    Returns summary statistics per dimension and overall.
    """
    if weights is None:
        weights = BCScore.DEFAULT_WEIGHTS

    composites = [s.composite_with_weights(weights) for s in scores]
    dims = {}
    for dim_name in ("identity", "goal", "abstraction", "style"):
        dim_scores = [getattr(s, dim_name).score for s in scores]
        dim_confs = [getattr(s, dim_name).confidence for s in scores]
        dims[dim_name] = {
            "mean": round(float(np.mean(dim_scores)), 4),
            "std": round(float(np.std(dim_scores)), 4),
            "min": round(float(np.min(dim_scores)), 4),
            "max": round(float(np.max(dim_scores)), 4),
            "mean_confidence": round(float(np.mean(dim_confs)), 4),
        }

    return {
        "n_evaluations": len(scores),
        "composite": {
            "mean": round(float(np.mean(composites)), 4),
            "std": round(float(np.std(composites)), 4),
            "min": round(float(np.min(composites)), 4),
            "max": round(float(np.max(composites)), 4),
        },
        "dimensions": dims,
    }
