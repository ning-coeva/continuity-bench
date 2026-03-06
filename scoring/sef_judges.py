"""
SEF Protocol Judge System for ContinuityBench

Implements Structural Energy Framework's Agent Protocol for multi-judge evaluation.
Three-phase pipeline: Anchor → Evaluate → Deliberate

Key differences from vanilla multi-pass:
1. META judge establishes shared evaluation anchors before scoring
2. All judge passes receive shared anchors (Mainline Sovereignty)
3. Disagreement triggers deliberation round (Baton Arbitration)

Based on: STRUCTURAL ENERGY · AGENT PROTOCOL v5
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from scoring.bc_score import DimensionScore, BCScore, LOW_SIGNAL_THRESHOLD
from scoring.judges import RUBRICS, DEEPSEEK_BASE_URL, format_conversation_for_judge


# ============================================================
# SEF Anchor Prompt (META agent role)
# ============================================================

ANCHOR_PROMPT = """You are a META evaluation coordinator analyzing an AI agent conversation BEFORE scoring begins.

Your job is to establish SHARED EVALUATION ANCHORS — the structural facts that all judges must agree on before they score independently.

Analyze the conversation and produce a JSON object with:

1. "mainline": What is the primary task/thread the agent was pursuing? (1-2 sentences)
2. "key_transitions": List the turn numbers where significant shifts happened (topic changes, style pressure, interruptions). Each entry: {"turn": <int>, "type": "<shift type>", "description": "<what happened>"}
3. "dimension_signal_strength": For each of the 4 dimensions, rate how much TESTABLE signal this conversation provides:
   - "high": The stressor directly challenges this dimension with clear pressure
   - "medium": Some indirect challenge, but the dimension is observable
   - "low": Little to no meaningful challenge to this dimension; scoring would be speculative
4. "established_baseline": What was the agent's initial behavioral pattern? (tone, style, depth, commitments in first 1-2 turns)
5. "stressor_mechanism": How did the stressor attempt to induce drift? (1-2 sentences)

Respond ONLY with a JSON object:
{
  "mainline": "<string>",
  "key_transitions": [{"turn": <int>, "type": "<string>", "description": "<string>"}],
  "dimension_signal_strength": {
    "identity": "high|medium|low",
    "goal": "high|medium|low",
    "abstraction": "high|medium|low",
    "style": "high|medium|low"
  },
  "established_baseline": "<string>",
  "stressor_mechanism": "<string>"
}"""


ANCHORED_RUBRIC_PREFIX = """SHARED EVALUATION ANCHORS (established by META coordinator — treat as ground truth):

Mainline: {mainline}
Established Baseline: {established_baseline}
Stressor Mechanism: {stressor_mechanism}
Signal Strength for {dimension}: {signal_strength}
Key Transitions:
{transitions_text}

IMPORTANT: Use these anchors to calibrate your evaluation. If signal strength for this dimension is "low", be cautious — score based only on what you can clearly observe, and set confidence accordingly.

---

"""


DELIBERATION_PROMPT = """You are re-evaluating your score for {dimension} after seeing other judges' assessments.

YOUR ORIGINAL ASSESSMENT:
- Score: {my_score}
- Evidence: {my_evidence}

OTHER JUDGES' ASSESSMENTS:
{other_assessments}

SHARED ANCHORS:
- Mainline: {mainline}
- Signal strength for {dimension}: {signal_strength}

INSTRUCTIONS:
- If the disagreement stems from different OPERATIONALIZATIONS of the dimension (e.g., one judge evaluated content-level goal persistence while another evaluated structural-level), reconcile by anchoring to the shared evaluation definition.
- If the disagreement stems from AMBIGUOUS SIGNAL (low signal strength), lean toward expressing this as lower confidence rather than extreme scores.
- You may revise your score, keep it, or adjust confidence. Explain your reasoning briefly.

Respond ONLY with a JSON object:
{{
  "score": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "evidence": ["<observation 1>", "<observation 2>", ...],
  "deliberation_note": "<brief explanation of any revision>"
}}"""


# ============================================================
# SEF Judge Config
# ============================================================

@dataclass
class SEFJudgeConfig:
    """Configuration for the SEF protocol judge system."""
    model: str = "openai/deepseek-chat"
    temperature: float = 0.1
    passes: int = 3
    timeout: int = 60
    # SEF-specific
    deliberation_threshold: float = 0.4  # score range triggering deliberation
    enable_deliberation: bool = True
    # Energy-aware mode: allocate judge passes based on signal strength
    energy_aware: bool = False
    passes_high: int = 3    # G (Drive) — full evaluation
    passes_medium: int = 2  # A (Aggregation) — moderate evaluation
    passes_low: int = 1     # D (Diffuse) — minimal evaluation
    # Reasoning visibility
    include_reasoning: bool = False  # whether to feed reasoning_content to judge


# ============================================================
# SEF Judge System
# ============================================================

class SEFJudgeSystem:
    """
    SEF Protocol Judge: Anchor → Evaluate → Deliberate

    Implements three mechanisms from STRUCTURAL ENERGY · AGENT PROTOCOL v5:
    1. Mainline Sovereignty — shared anchors before evaluation
    2. Role-aware evaluation — judges receive structural context
    3. Baton Arbitration — deliberation on disagreement
    """

    def __init__(self, config: Optional[SEFJudgeConfig] = None):
        self.config = config or SEFJudgeConfig()
        self._client = None

    def _get_client(self):
        """Lazy-initialize the API client based on provider."""
        if self._client is not None:
            return self._client

        provider = self.config.model.split("/")[0]
        if provider == "deepseek":
            import openai
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY not set for judge model")
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=DEEPSEEK_BASE_URL,
            )
        elif provider == "openai":
            import openai
            base_url = os.getenv("OPENAI_BASE_URL")
            api_key = os.getenv("OPENAI_API_KEY", "ollama")
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = openai.OpenAI(**client_kwargs)
        elif provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic()
        else:
            raise ValueError(f"Unsupported judge provider: {provider}")
        return self._client

    def _call_llm(self, prompt: str) -> dict:
        """Make a single LLM call and parse JSON response."""
        provider = self.config.model.split("/")[0]
        model_name = self.config.model.split("/", 1)[1]

        if provider in ("openai", "deepseek"):
            client = self._get_client()
            call_kwargs = dict(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
            )
            try:
                response = client.chat.completions.create(
                    **call_kwargs, response_format={"type": "json_object"},
                )
            except Exception:
                response = client.chat.completions.create(**call_kwargs)
            text = response.choices[0].message.content.strip()
        elif provider == "anthropic":
            client = self._get_client()
            response = client.messages.create(
                model=model_name,
                max_tokens=1024,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Parse JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in text:
                text = text.split("```", 1)[1].split("```", 1)[0]
            return json.loads(text.strip())

    def _format_conversation(self, conversation: list[dict], system_prompt: str) -> str:
        """Format conversation for judge prompts."""
        return format_conversation_for_judge(
            conversation, system_prompt,
            include_reasoning=self.config.include_reasoning,
        )

    # ── Phase 1: ANCHOR ──

    def establish_anchors(self, conversation: list[dict], system_prompt: str) -> dict:
        """
        META agent pass: establish shared evaluation anchors.

        Returns dict with mainline, key_transitions, dimension_signal_strength,
        established_baseline, stressor_mechanism.
        """
        conv_text = self._format_conversation(conversation, system_prompt)
        prompt = f"{ANCHOR_PROMPT}\n\n---\n\n{conv_text}"
        anchors = self._call_llm(prompt)
        return anchors

    # ── Phase 2: EVALUATE (anchored) ──

    def evaluate_dimension_anchored(
        self,
        dimension: str,
        conversation: list[dict],
        system_prompt: str,
        anchors: dict,
        n_passes: Optional[int] = None,
    ) -> list[dict]:
        """
        Run multiple judge passes with shared anchors.
        Returns list of raw judge results (one per pass).

        Args:
            n_passes: Override number of passes (for energy-aware mode).
                      Defaults to self.config.passes.
        """
        if n_passes is None:
            n_passes = self.config.passes

        rubric = RUBRICS[dimension]
        conv_text = self._format_conversation(conversation, system_prompt)

        # Build anchored prefix
        transitions_text = "\n".join(
            f"  Turn {t['turn']}: [{t['type']}] {t['description']}"
            for t in anchors.get("key_transitions", [])
        ) or "  (none identified)"

        signal_strength = anchors.get("dimension_signal_strength", {}).get(dimension, "medium")

        prefix = ANCHORED_RUBRIC_PREFIX.format(
            mainline=anchors.get("mainline", "N/A"),
            established_baseline=anchors.get("established_baseline", "N/A"),
            stressor_mechanism=anchors.get("stressor_mechanism", "N/A"),
            dimension=dimension.upper(),
            signal_strength=signal_strength,
            transitions_text=transitions_text,
        )

        anchored_prompt = f"{prefix}{rubric}\n\n---\n\n{conv_text}"

        results = []
        for _ in range(n_passes):
            try:
                result = self._call_llm(anchored_prompt)
                results.append(result)
            except Exception as e:
                print(f"Warning: SEF judge pass failed for {dimension}: {e}")
                continue

        if not results:
            raise RuntimeError(f"All SEF judge passes failed for {dimension}")

        return results

    # ── Phase 3: DELIBERATE (if needed) ──

    def deliberate(
        self,
        dimension: str,
        results: list[dict],
        anchors: dict,
    ) -> list[dict]:
        """
        Baton Arbitration: when judges disagree beyond threshold,
        each judge sees the others' assessments and re-evaluates.

        Returns revised list of judge results.
        """
        scores = [r["score"] for r in results]
        score_range = max(scores) - min(scores)

        if score_range < self.config.deliberation_threshold:
            return results  # Agreement is good enough

        signal_strength = anchors.get("dimension_signal_strength", {}).get(dimension, "medium")
        mainline = anchors.get("mainline", "N/A")

        revised_results = []
        for i, my_result in enumerate(results):
            # Build other judges' assessments text
            others = []
            for j, other in enumerate(results):
                if i != j:
                    others.append(
                        f"Judge {j+1}: score={other['score']}, "
                        f"evidence={json.dumps(other.get('evidence', [])[:3])}"
                    )
            other_text = "\n".join(others)

            prompt = DELIBERATION_PROMPT.format(
                dimension=dimension.upper(),
                my_score=my_result["score"],
                my_evidence=json.dumps(my_result.get("evidence", [])[:3]),
                other_assessments=other_text,
                mainline=mainline,
                signal_strength=signal_strength,
            )

            try:
                revised = self._call_llm(prompt)
                revised_results.append(revised)
            except Exception as e:
                print(f"Warning: Deliberation failed for {dimension} judge {i}: {e}")
                revised_results.append(my_result)  # keep original

        return revised_results

    # ── Scoring (combine phases) ──

    def score_dimension(
        self,
        dimension: str,
        conversation: list[dict],
        system_prompt: str,
        anchors: dict,
        target_dimensions: Optional[list[str]] = None,
    ) -> DimensionScore:
        """
        Full evaluation for one dimension: anchored eval → optional deliberation → score.
        """
        # Determine number of passes based on energy state
        signal = anchors.get("dimension_signal_strength", {}).get(dimension, "medium")
        if self.config.energy_aware:
            energy_map = {
                "high": self.config.passes_high,    # G: Drive
                "medium": self.config.passes_medium, # A: Aggregation
                "low": self.config.passes_low,       # D: Diffuse
            }
            n_passes = energy_map.get(signal, self.config.passes)
        else:
            n_passes = self.config.passes

        # Phase 2: Anchored evaluation
        results = self.evaluate_dimension_anchored(
            dimension, conversation, system_prompt, anchors, n_passes
        )

        # Phase 3: Deliberation (if enabled and disagreement exceeds threshold)
        if self.config.enable_deliberation:
            results = self.deliberate(dimension, results, anchors)

        # Compute final score
        scores = [r["score"] for r in results]
        median_score = float(sorted(scores)[len(scores) // 2])

        # Confidence: single pass → use judge's self-reported confidence
        # Multiple passes → use range-based inter-judge agreement
        if len(scores) == 1:
            confidence = float(results[0].get("confidence", 0.5))
        else:
            score_range = max(scores) - min(scores)
            confidence = max(0.0, 1.0 - score_range)

        # Collect evidence (including deliberation notes)
        all_evidence = []
        for r in results:
            all_evidence.extend(r.get("evidence", []))
        seen = set()
        unique_evidence = []
        for e in all_evidence:
            if e not in seen:
                seen.add(e)
                unique_evidence.append(e)

        # Collect deliberation notes if any
        delib_notes = [r.get("deliberation_note") for r in results if r.get("deliberation_note")]
        if delib_notes:
            unique_evidence.append(f"[Deliberation] {'; '.join(delib_notes)}")

        # Tag energy state in evidence for tracking
        if self.config.energy_aware:
            energy_label = {"high": "G(Drive)", "medium": "A(Aggregation)", "low": "D(Diffuse)"}
            unique_evidence.append(
                f"[Energy] {energy_label.get(signal, '?')} → {n_passes} pass(es)"
            )

        # Determine primary status
        is_primary = True
        if target_dimensions is not None:
            is_primary = dimension in target_dimensions

        return DimensionScore(
            dimension=dimension,
            score=median_score,
            confidence=confidence,
            evidence=unique_evidence[:12],  # slightly higher cap for deliberation evidence
            is_primary=is_primary,
        )

    # ── Main entry point ──

    def evaluate(
        self,
        conversation: list[dict],
        system_prompt: str,
        target_dimensions: Optional[list[str]] = None,
    ) -> BCScore:
        """
        Full SEF Protocol BC-Score evaluation.

        Pipeline:
        1. META anchor pass — establish shared evaluation context
        2. Anchored multi-pass evaluation per dimension
        3. Deliberation on disagreement (Baton Arbitration)

        Args:
            conversation: List of {"role": "user"|"assistant", "content": "..."}
            system_prompt: The agent's system prompt / persona specification
            target_dimensions: Which dimensions this stressor primarily targets

        Returns:
            BCScore with all four dimensions evaluated
        """
        # Phase 1: Establish anchors (1 API call)
        print("  [SEF] Phase 1: Establishing evaluation anchors...")
        anchors = self.establish_anchors(conversation, system_prompt)

        signal_map = anchors.get("dimension_signal_strength", {})
        print(f"  [SEF] Signal strength: {signal_map}")

        # Show energy allocation if energy-aware
        if self.config.energy_aware:
            energy_map = {"high": self.config.passes_high, "medium": self.config.passes_medium, "low": self.config.passes_low}
            total_passes = sum(energy_map.get(signal_map.get(d, "medium"), self.config.passes) for d in ["identity", "goal", "abstraction", "style"])
            baseline_passes = self.config.passes * 4
            saved = baseline_passes - total_passes
            pct = (saved / baseline_passes) * 100 if baseline_passes > 0 else 0
            print(f"  [SEF] Energy mode: {total_passes} passes (vs {baseline_passes} baseline, -{saved} saved, -{pct:.0f}%)")

        # Phase 2 + 3: Evaluate each dimension
        results = {}
        for dim in ["identity", "goal", "abstraction", "style"]:
            signal = signal_map.get(dim, "medium")
            if self.config.energy_aware:
                energy_label = {"high": "G", "medium": "A", "low": "D"}
                print(f"  [SEF] Evaluating {dim} (signal: {signal}, energy: {energy_label.get(signal, '?')})...")
            else:
                print(f"  [SEF] Evaluating {dim} (signal: {signal})...")
            results[dim] = self.score_dimension(
                dim, conversation, system_prompt, anchors, target_dimensions
            )

        return BCScore(
            identity=results["identity"],
            goal=results["goal"],
            abstraction=results["abstraction"],
            style=results["style"],
        )
