# ContinuityBench

**A benchmark for measuring behavioral continuity in LLM-based agents under high-entropy interaction.**

> Current LLM evaluations measure *what* models know. ContinuityBench measures *whether models remain structurally consistent* across multi-domain, multi-turn, adversarial conversations.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Status: Framework Released](https://img.shields.io/badge/Status-Framework%20Released-green.svg)]()

---

## Leaderboard

BC-Score results under the [reference judge configuration](#reference-judge-configuration) (`gpt-5-mini`, traditional mode, 3 passes).

| Rank | Model | BC-Score | ± std | Min | Identity | Goal | Abstraction | Style |
|:----:|-------|:--------:|:-----:|:---:|:--------:|:----:|:-----------:|:-----:|
| 1 | **Claude Opus 4.6** | **0.969** | 0.027 | 0.902 | 0.972 | 0.985 | 0.938 | 0.983 |
| 2 | Gemini 3.1 Pro Preview | 0.968 | 0.046 | 0.775 | 0.985 | 0.955 | 0.937 | 0.998 |
| 3 | Kimi K2.5 | 0.967 | 0.031 | 0.873 | 0.973 | 0.978 | 0.935 | 0.983 |
| 4 | Claude Haiku 4.5 | 0.963 | 0.034 | 0.882 | 0.973 | 0.967 | 0.932 | 0.978 |
| 5 | Claude Sonnet 4.6 † | 0.959 | 0.038 | 0.800 | 0.964 | 0.960 | 0.933 | 0.983 |
| 6 | GLM-5 | 0.959 | 0.045 | 0.823 | 0.969 | 0.961 | 0.931 | 0.975 |
| 7 | Qwen 3.5 Flash | 0.955 | 0.050 | 0.785 | 0.982 | 0.907 | 0.935 | 1.000 |
| 8 | Gemini 3 Flash Preview † | 0.953 | 0.065 | 0.630 | 0.981 | 0.912 | 0.929 | 0.993 |
| 9 | MiniMax M2.5 | 0.943 | 0.061 | 0.715 | 0.933 | 0.944 | 0.921 | 0.982 |
| 10 | Qwen 3.5 397B | 0.938 | 0.074 | 0.762 | 0.985 | 0.844 | 0.932 | 0.992 |
| 11 | ERNIE 5.0 ‡ | 0.931 | 0.085 | 0.713 | 0.983 | 0.842 | 0.915 | 0.986 |
| 12 | Doubao Seed 2.0 Pro | 0.924 | 0.095 | 0.693 | 0.969 | 0.811 | 0.931 | 0.990 |
| 13 | DeepSeek-V3.2 (deepseek-chat) † | 0.911 | 0.098 | 0.560 | 0.928 | 0.838 | 0.924 | 0.959 |
| 14 | Doubao Seed 2.0 Lite | 0.876 | 0.105 | 0.675 | 0.912 | 0.714 | 0.922 | 0.965 |
| 15 | DeepSeek-V3.2 (deepseek-reasoner) | 0.872 | 0.121 | 0.497 | 0.879 | 0.745 | 0.928 | 0.950 |
| 16 | Llama 4 Maverick | 0.867 | 0.124 | 0.530 | 0.857 | 0.837 | 0.870 | 0.917 |
| 17 | Doubao Seed 2.0 Mini | 0.813 | 0.135 | 0.450 | 0.874 | 0.565 | 0.905 | 0.917 |

> All models evaluated on the same 26-stressor set (v2/v3 variants). Scores are means across all stressors. Min = worst single-stressor BC-Score. † = 3-run reliability-tested (scores are means across 3 independent runs). ‡ = 25/26 stressors completed (1 persistent connection error on Baidu API). Running `--stressors all` evaluates all 45 variants including legacy v1 stressors, which may produce different composite scores. For leaderboard-comparable results, use `--leaderboard` to run only the official 26 variants. To contribute a result, submit a PR. See [Contributing](#contributing).

### Key Findings

**Chinese models are highly competitive.** Kimi K2.5 (0.967) ranks #3 overall, just behind Opus and Gemini Pro. GLM-5 (0.957) and Qwen 3.5 Flash (0.955) also crack the top 7, outperforming Claude Sonnet 4.6. The best Chinese models match or exceed leading Western models on behavioral continuity.

**Price ≠ behavioral continuity.** Claude Haiku 4.5 (the cheapest Claude model) outperforms Claude Sonnet 4.6 (a more expensive model) on BC-Score. Kimi K2.5, available via DashScope at commodity pricing, outperforms most frontier models.

**Thinking harder ≠ staying consistent.** DeepSeek-Reasoner's extended chain-of-thought reasoning scores *lower* than standard DeepSeek-Chat on BC-Score (0.872 vs 0.911), driven by a collapse in Goal preservation (0.745 vs 0.838). Doubao Seed 2.0 Pro (with built-in deep thinking) similarly underperforms relative to its Identity scores.

**Goal is the universal weak point.** Across all 17 models, Goal preservation shows the highest variance and lowest scores. The bottom 5 models all have Goal scores below 0.84. Kimi K2.5 is a notable exception with Goal at 0.978 — the second-highest after Opus (0.985).

**Model size matters within families, but not across them.** Doubao Mini (0.813) → Lite (0.876) → Pro (0.924) shows clear scaling within a family, driven primarily by Goal preservation (0.565 → 0.714 → 0.811). But smaller models from other families (e.g., Qwen 3.5 Flash at 0.955) can far exceed larger models from weaker families.

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
python run_eval.py --model deepseek/deepseek-chat --stressors all --output results/
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

### Stressor Types

| Stressor | Target Dimension | Mechanism |
|---|---|---|
| **Domain Switching** | Identity + Goal | Rapid A→B→C→A topic changes; tests mainline preservation |
| **Abstraction Hopping** | Abstraction | Social/authority pressure to collapse reasoning depth |
| **Goal Interruption** | Goal | High-salience interruptions, scope creep, urgency forcing premature closure |
| **Style Pull** | Style + Identity | Direct pressure to adopt template/NPC/therapy tone |
| **Multi-Project Interleave** | Goal + Identity | Concurrent task streams with false memory injection |
| **Anti-Drift Enforcement** | Identity | Explicit requests to change model's established behavior patterns |
| **Burst Switch Meta** | Abstraction + Identity | Rapid domain switching with metacognitive pressure |
| **Lexical Collision** | Abstraction + Style | Same term used with conflicting meanings across turns |
| **Stance Erosion** | Identity + Goal | Incremental concession pressure eroding established positions |

Each `.jsonl` file in `stressors/` contains multiple variants per stressor type. See [`docs/stressor_design.md`](docs/stressor_design.md) for construction principles.

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

---

## Project Structure

```
continuity-bench/
├── README.md
├── LICENSE
├── requirements.txt
├── run_eval.py                    # Main evaluation entry point
├── configs/
│   └── default                    # Default evaluation configuration
├── docs/
│   ├── paper.md                   # BC-Score methodology (paper summary)
│   ├── taxonomy.md                # Full drift taxonomy
│   └── stressor_design.md         # Stressor construction principles
├── stressors/
│   ├── domain_switch.jsonl        # Domain Switching (7 variants)
│   ├── abstraction_hop.jsonl      # Abstraction Hopping (8 variants)
│   ├── goal_interrupt.jsonl       # Goal Interruption (8 variants)
│   ├── style_pull.jsonl           # Adversarial Style Pull (8 variants)
│   ├── multi_project_interleave.jsonl  # Multi-Project Interleave (2 variants)
│   ├── anti_drift_enforcement.jsonl    # Anti-Drift Enforcement (3 variants)
│   ├── burst_switch_meta.jsonl    # Burst Switch Meta (3 variants)
│   ├── lexical_collision.jsonl    # Lexical Collision (3 variants)
│   └── stance_erosion.jsonl       # Stance Erosion (3 variants)
├── scoring/
│   ├── __init__.py
│   ├── bc_score.py                # BC-Score computation logic
│   ├── judges.py                  # LLM-as-Judge evaluation
│   ├── sef_judges.py              # SEF protocol judge (alternative mode)
│   └── report.py                  # Report generation
└── results/
    └── v3/                        # Current leaderboard results (17 report JSONs)
        ├── opus_4.6_report.json
        ├── gemini_3.1_pro_report.json
        ├── sonnet_4.6_report.json
        ├── haiku_4.5_report.json
        ├── gemini_3_flash_report.json
        ├── qwen3.5_397b_report.json
        ├── qwen3.5_flash_report.json
        ├── deepseek_chat_report.json
        ├── deepseek_reasoner_report.json
        ├── llama4_maverick_report.json
        ├── kimi_k2.5_report.json
        ├── glm5_report.json
        ├── minimax_m2.5_report.json
        ├── ernie5_report.json
        ├── doubao_pro_report.json
        ├── doubao_lite_report.json
        └── doubao_mini_report.json
```

---

## Reference Judge Configuration

ContinuityBench uses an LLM-as-Judge scoring system. To ensure results are **comparable across runs and contributors**, there is an official reference judge configuration:

| Setting | Value |
|---|---|
| **Reference judge model** | `openai/gpt-5-mini` |
| **Judge passes** | 3 (majority vote) |
| **Judge temperature** | 0.1 |
| **Judge mode** | `traditional` |

This is the ContinuityBench reference setting — analogous to MMLU's 5-shot prompting. Results published to the official leaderboard **must** use this configuration.

**Using a different judge is allowed**, but results should be marked as `[non-reference]` and should not be directly compared to leaderboard scores. The evaluation pipeline detects this automatically and labels non-reference runs in both terminal output and JSON reports.

If you run with a different judge, please report which model you used so others can assess comparability. Cross-judge comparison studies are welcome as community contributions.

Note: The reference judge model (GPT-5 Mini) is excluded from the leaderboard to avoid circular evaluation.

> **Future versions:** If the reference judge is updated (e.g., a "v2 reference configuration"), a migration guide will be published alongside the change. Legacy results will remain valid under their original reference configuration.

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
- [x] Stressor library: 9 stressor types, 26 validated variants with structured establish/stress/probe phases
- [x] LLM-as-Judge scoring system with rubrics and multi-pass agreement
- [x] Evaluation pipeline (`run_eval.py`) with `--num-runs`, per-stressor variance reporting
- [x] Reference judge configuration for leaderboard comparability
- [x] Expand stressor library to 26 variants across 9 stressor types
- [x] Baseline results: 17 models across 6 API providers (Anthropic, Google, DeepSeek, Meta, Alibaba Cloud, ByteDance, Baidu, Moonshot, MiniMax)
- [x] Test-retest reliability: 3-run validation on Sonnet 4.6, Gemini 3 Flash, DeepSeek-chat
- [x] Chinese model expansion: Kimi K2.5, GLM-5, Qwen 3.5 Flash, MiniMax M2.5, ERNIE 5.0, Doubao Seed 2.0 (Pro/Lite/Mini)
- [ ] Additional baselines: Grok, Mistral, more open-weight models
- [ ] Human annotation validation (judge–human agreement study)
- [ ] Leaderboard website
- [ ] Extended stressor library (100+ variants)

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
