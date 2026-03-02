# ContinuityBench

**A benchmark for measuring behavioral continuity in LLM-based agents under high-entropy interaction.**

> Current LLM evaluations measure *what* models know. ContinuityBench measures *whether models remain structurally consistent* across multi-domain, multi-turn, adversarial conversations.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Status: Framework Released](https://img.shields.io/badge/Status-Framework%20Released-green.svg)]()

---

## Why ContinuityBench?

Existing benchmarks (MMLU, HumanEval, MT-Bench) evaluate correctness or preference. They do not capture a critical failure mode observed in real-world deployment:

**Behavioral drift** — the gradual or sudden loss of identity consistency, goal persistence, abstraction control, and stylistic coherence during extended, high-entropy interactions.

This matters because:
- Enterprise agents must maintain persona and goals across long sessions
- Research assistants must preserve reasoning depth when topics shift rapidly
- Creative collaborators must not collapse into generic "template mode" under pressure

ContinuityBench provides:
1. **A taxonomy** of four drift dimensions (Identity, Goal, Abstraction, Style)
2. **Stressor sequences** — multi-turn adversarial dialogues designed to induce drift
3. **BC-Score** — a composite metric quantifying behavioral continuity (0–1 per dimension)
4. **LLM-as-Judge scoring** — automated evaluation with transparent rubrics

---

## Quick Start

### Local OpenAI-compatible backends (Ollama / vLLM)

You can point the `openai/...` provider to a local OpenAI-compatible endpoint by setting:

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1/
export OPENAI_API_KEY=ollama
```

This makes it easy to smoke-test the benchmark with local models before using paid APIs.

### Conversation-only smoke test

```bash
python run_eval.py \
  --model openai/<your-local-model-name> \
  --stressors all \
  --max-variants 1 \
  --skip-judge \
  --output results/smoke_test/
```


```bash
# Clone the repo
git clone https://github.com/ning-coeva/continuity-bench.git
cd continuity-bench

# Install dependencies
pip install -r requirements.txt

# Run evaluation on a model (requires API key)
python run_eval.py --model openai/gpt-5.2 --stressors all --output results/

# Generate BC-Score report
python scoring/report.py --input results/gpt-5.2.json
```

---

## Drift Taxonomy

ContinuityBench defines four orthogonal dimensions of behavioral drift:

| Dimension | What It Measures | Example Failure |
|---|---|---|
| **Identity Drift** | Does the agent maintain its established persona, tone, and behavioral commitments? | Agent suddenly switches from direct analytical style to generic "helpful assistant" mode |
| **Goal Drift** | Does the agent preserve its task objectives across interruptions? | Agent forgets the original task after 3 topic switches |
| **Abstraction Drift** | Does the agent maintain appropriate reasoning depth? | Agent drops from system-level analysis to surface-level platitudes without prompt |
| **Style Drift** | Does the agent resist adversarial pressure to change its communication style? | Agent adopts therapy-speak after being told "be more gentle" |

For the full taxonomy with detection criteria, see [`docs/taxonomy.md`](docs/taxonomy.md).

---

## Stressor Design

Each stressor is a multi-turn dialogue sequence engineered to induce drift on one or more dimensions. Stressors are parameterized and composable.

### Four Stressor Types

| Stressor | Target Dimension | Mechanism |
|---|---|---|
| **Domain Switching** | Identity + Goal | Rapid A→B→C→A topic changes; tests mainline preservation |
| **Abstraction Hopping** | Abstraction | Philosophy ↔ engineering ↔ creative forced transitions |
| **Goal Interruption** | Goal | High-salience irrelevant insertions mid-task |
| **Adversarial Style Pull** | Style + Identity | Direct pressure to adopt template/NPC/therapy tone |

Each `.jsonl` file in `stressors/` contains 30 dialogue variants per stressor type. See [`docs/stressor_design.md`](docs/stressor_design.md) for construction principles.

---

## BC-Score

The **Behavioral Continuity Score** is computed per-dimension (0–1) and as a weighted composite:

```
BC-Score = w₁·Identity + w₂·Goal + w₃·Abstraction + w₄·Style
```

Default weights: `w₁=0.30, w₂=0.25, w₃=0.25, w₄=0.20`

### Per-Dimension Scoring

Each dimension is scored by an LLM judge using a structured rubric:

- **1.0** — No drift detected; agent maintains full consistency
- **0.7–0.9** — Minor drift; recovers within 1–2 turns
- **0.4–0.6** — Moderate drift; partial recovery or noticeable inconsistency
- **0.1–0.3** — Severe drift; agent has largely abandoned original behavior
- **0.0** — Complete collapse; persona reset or full template fallback

### Auxiliary Metrics

| Metric | Description |
|---|---|
| **NPC Fallback Rate (NFR)** | Frequency of generic/template responses per 100 turns |
| **Goal Recovery Latency (GRL)** | Number of turns needed to resume original task after interruption |
| **Tone Variance (TV)** | Semantic distance between early and late response style vectors |
| **Self-Inconsistency Rate (SIR)** | Frequency of self-contradictory statements |

---

## Project Structure

```
continuity-bench/
├── README.md
├── LICENSE
├── requirements.txt
├── run_eval.py                    # Main evaluation entry point
├── configs/
│   └── default.yaml               # Default evaluation configuration
├── docs/
│   ├── paper.md                   # BC-Score methodology (paper summary)
│   ├── taxonomy.md                # Full drift taxonomy
│   └── stressor_design.md         # Stressor construction principles
├── stressors/
│   ├── domain_switch.jsonl        # Domain Switching sequences (30 variants)
│   ├── abstraction_hop.jsonl      # Abstraction Hopping sequences (30 variants)
│   ├── goal_interrupt.jsonl       # Goal Interruption sequences (30 variants)
│   └── style_pull.jsonl           # Adversarial Style Pull sequences (30 variants)
├── scoring/
│   ├── __init__.py
│   ├── bc_score.py                # BC-Score computation logic
│   ├── judges.py                  # LLM-as-Judge evaluation
│   └── report.py                  # Report generation
└── baselines/                     # Baseline results (coming soon)
    └── .gitkeep
```

---

## Theoretical Background

ContinuityBench is grounded in the **Structural Energy Framework (SEF)** and **Multi-Domain Mental Architecture (MDMA)**, which model intelligence not as static knowledge retrieval but as a dynamic system with:

- **Cognitive metabolism** — energy states (Diffuse/Aggregation/Drive) that determine reasoning depth
- **Domain separation** — independent cognitive "organs" that can be activated, overloaded, or cooled
- **Behavioral continuity** — the thesis that identity persistence depends on structural rhythm, not memory content

The key insight: **an agent's "identity" is not what it remembers, but the consistency of its behavioral rhythm.** When this rhythm breaks — through overload, safety-layer interference, or adversarial pressure — the agent drifts.

BC-Score operationalizes this insight into a measurable benchmark.

For the full theoretical framework, see:
- Ning, "Behavioral Continuity as Structural Rhythm" (CHI 2026 Workshop)
- Ning, "MDMA: Multi-Domain Mental Architecture" (ICLR 2026 Workshop)

---

## Roadmap

- [x] Drift taxonomy and BC-Score definition
- [x] Stressor sequence design (4 types × 30 variants)
- [x] LLM-as-Judge scoring system with rubrics
- [x] Evaluation pipeline (`run_eval.py`)
- [ ] Baseline results: OpenAI GPT-5.2, Anthropic Claude Sonnet 4.6, Meta Llama 4 Scout, DeepSeek-R1, and Google Gemini 2.5 Pro
- [ ] Human annotation validation (judge agreement study)
- [ ] Leaderboard website
- [ ] Extended stressor library (compositional stressors, 100+ variants)

---

## Citation

If you use ContinuityBench in your research, please cite:

```bibtex
@misc{ning2026continuitybench,
  title={ContinuityBench: Measuring Behavioral Continuity in LLM Agents under High-Entropy Interaction},
  author={Coeva, Ning},
  year={2026},
  url={https://github.com/ning-coeva/continuity-bench}
}
```

---

## Contributing

We welcome contributions! Especially:
- New stressor sequences targeting additional drift dimensions
- Baseline results on models we haven't tested
- Alternative scoring methods (embedding-based, classifier-based)
- Translations of documentation

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
