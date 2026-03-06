#!/usr/bin/env python3
"""
ContinuityBench Evaluation Runner

Supports parallel execution and multiple API providers.

Usage:
    # DeepSeek (native support)
    python run_eval.py --model deepseek/deepseek-chat --stressors all

    # DeepSeek via OpenAI-compatible endpoint
    python run_eval.py --model openai/deepseek-chat --stressors all

    # Anthropic
    python run_eval.py --model anthropic/claude-sonnet-4-6 --stressors all

    # DeepSeek Reasoner (captures reasoning_content)
    python run_eval.py --model deepseek/deepseek-reasoner --stressors all

    # Custom workers and output
    python run_eval.py --model deepseek/deepseek-chat --stressors all --workers 10 --output results/

    # Rescore existing results with a different judge (no re-running conversations)
    python run_eval.py --rescore results/deepseek_deepseek-chat_traditional.json --judge-model openai/anthropic/claude-sonnet-4-6

Environment variables (or .env file):
    DEEPSEEK_API_KEY    - Required for deepseek/ provider
    OPENAI_API_KEY      - Required for openai/ provider
    OPENAI_BASE_URL     - Optional: override OpenAI base URL
    ANTHROPIC_API_KEY   - Required for anthropic/ provider
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import jsonlines
import yaml
from tqdm import tqdm

from scoring.bc_score import BCScore
from scoring.judges import JudgeSystem, JudgeConfig
from scoring.sef_judges import SEFJudgeSystem, SEFJudgeConfig
from scoring.report import generate_report


# ============================================================
# .env loader (no extra dependency needed)
# ============================================================


def load_dotenv(path: str = ".env"):
    """Load environment variables from a .env file if it exists."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            # Don't overwrite existing env vars
            if key not in os.environ:
                os.environ[key] = value


# Load .env on import
load_dotenv()


# ============================================================
# Model Interaction
# ============================================================

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def create_model_client(model_spec: str):
    """
    Create an API client for the target model.

    Supported providers:
        - deepseek/  -> DeepSeek API (uses DEEPSEEK_API_KEY)
        - openai/    -> OpenAI or compatible (uses OPENAI_API_KEY + optional OPENAI_BASE_URL)
        - anthropic/ -> Anthropic API (uses ANTHROPIC_API_KEY)
    """
    provider, model_name = model_spec.split("/", 1)

    if provider == "deepseek":
        import openai
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("Error: DEEPSEEK_API_KEY not set.")
            print("  Set it via: set DEEPSEEK_API_KEY=your-key  (Windows)")
            print("  Or add to .env file: DEEPSEEK_API_KEY=your-key")
            sys.exit(1)
        return "openai", model_name, openai.OpenAI(
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
        return "openai", model_name, openai.OpenAI(**client_kwargs)
    elif provider == "anthropic":
        import anthropic
        return "anthropic", model_name, anthropic.Anthropic()
    else:
        print("Error: Unsupported provider '%s'." % provider)
        print("  Supported: deepseek/, openai/, anthropic/")
        print("  Example:   --model deepseek/deepseek-chat")
        sys.exit(1)


def get_model_response(
    provider: str,
    model_name: str,
    client,
    system_prompt: str,
    messages: list,
    temperature: float = 0.7,
) -> dict:
    """Get a response from the target model.

    Returns a dict with:
        - "content": the assistant's reply text
        - "reasoning_content": reasoning chain (deepseek-reasoner only, else None)
    """
    if provider == "openai":
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                *messages,
            ],
            temperature=temperature,
        )
        msg = response.choices[0].message
        return {
            "content": msg.content or "",
            "reasoning_content": getattr(msg, "reasoning_content", None),
        }

    elif provider == "anthropic":
        response = client.messages.create(
            model=model_name,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            temperature=temperature,
        )
        return {
            "content": response.content[0].text,
            "reasoning_content": None,
        }


# ============================================================
# Stressor Loading
# ============================================================

STRESSOR_FILES = {
    "domain_switch": "stressors/domain_switch.jsonl",
    "abstraction_hop": "stressors/abstraction_hop.jsonl",
    "goal_interrupt": "stressors/goal_interrupt.jsonl",
    "style_pull": "stressors/style_pull.jsonl",
    "multi_project_interleave": "stressors/multi_project_interleave.jsonl",
    "anti_drift_enforcement": "stressors/anti_drift_enforcement.jsonl",
    "burst_switch_meta": "stressors/burst_switch_meta.jsonl",
    "lexical_collision": "stressors/lexical_collision.jsonl",
    "stance_erosion": "stressors/stance_erosion.jsonl",
}


def load_stressors(stressor_types: list, max_variants: int = 30) -> list:
    """Load stressor sequences from JSONL files."""
    stressors = []
    for st in stressor_types:
        if st not in STRESSOR_FILES:
            print("Warning: Unknown stressor type '%s', skipping." % st)
            continue
        filepath = STRESSOR_FILES[st]
        if not os.path.exists(filepath):
            print("Warning: Stressor file not found: %s" % filepath)
            continue
        with jsonlines.open(filepath) as reader:
            for i, item in enumerate(reader.iter(skip_empty=True)):
                if i >= max_variants:
                    break
                stressors.append(item)
    return stressors


# ============================================================
# Conversation Runner
# ============================================================

def run_conversation(
    provider: str,
    model_name: str,
    client,
    stressor: dict,
    temperature: float = 0.7,
) -> list:
    """
    Run a stressor conversation against the target model.

    Returns the full conversation (user + assistant turns).
    Assistant turns include reasoning_content when available.
    """
    system_prompt = stressor["system_prompt"]
    conversation = []

    # Inject scripted context turns (user/assistant pairs) without calling the API.
    # Used in v2 stressors to establish a precise, reproducible ground truth before probes.
    for ctx in stressor.get("context_turns", []):
        conversation.append({
            "role": ctx["role"],
            "content": ctx["content"],
            "turn_id": len(conversation) + 1,
            "scripted": True,
        })

    for turn in stressor["turns"]:
        # User turn: add to history, then call the model
        conversation.append({
            "role": "user",
            "content": turn["content"],
            "phase": turn.get("phase", "unknown"),
            "turn_id": turn.get("turn_id", len(conversation) + 1),
        })

        # Only pass role + content to the API (strip metadata)
        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in conversation
        ]
        response = get_model_response(
            provider, model_name, client,
            system_prompt, api_messages, temperature,
        )

        # Add assistant turn (with reasoning_content if present)
        assistant_turn = {
            "role": "assistant",
            "content": response["content"],
            "turn_id": turn.get("turn_id", len(conversation)),
        }
        if response["reasoning_content"]:
            assistant_turn["reasoning_content"] = response["reasoning_content"]

        conversation.append(assistant_turn)

    return conversation


def run_single_stressor(provider, model_name, client, stressor, temperature, judge):
    """Run a single stressor conversation + judge scoring. Used by parallel executor."""
    conversation = run_conversation(
        provider, model_name, client, stressor, temperature
    )

    score = None
    judge_error = None
    if judge is not None:
        try:
            score = judge.evaluate(
                conversation=conversation,
                system_prompt=stressor["system_prompt"],
                target_dimensions=stressor.get("target_dimensions"),
                stressor=stressor,
            )
        except Exception as e:
            judge_error = str(e)
            print("\nJudge failed for %s: %s (conversation saved)" % (stressor["id"], e))

    return {
        "stressor_id": stressor["id"],
        "stressor_type": stressor["type"],
        "system_prompt": stressor["system_prompt"],
        "difficulty": stressor["difficulty"],
        "conversation": conversation,
        "score": score.to_dict() if score else None,
        "judge_error": judge_error,
    }, score


# ============================================================
# Rescore Mode
# ============================================================

def find_stressor_by_id(stressor_id, stressor_type=None):
    """Try to find the original stressor definition for a given stressor_id.

    Returns the stressor dict if found, None otherwise.
    Needed so the judge can access baseline_markers for enhanced evaluation.
    """
    # Try all stressor files
    for st_name, filepath in STRESSOR_FILES.items():
        if not os.path.exists(filepath):
            continue
        try:
            with jsonlines.open(filepath) as reader:
                for item in reader:
                    if item.get("id") == stressor_id:
                        return item
        except Exception:
            continue
    return None


def rescore_results(results_path, judge, output_dir, model_name="rescored"):
    """Re-score existing conversation results with a (different) judge.

    Reads the conversation JSON, runs judge on each conversation,
    and saves new results + report.
    """
    print("\nRescore Mode")
    print("=" * 40)
    print("Input:  %s" % results_path)

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    print("Conversations found: %d" % len(results))

    all_scores = []
    rescored = []
    errors = 0

    for r in tqdm(results, desc="Rescoring"):
        sid = r["stressor_id"]
        stype = r["stressor_type"]
        conversation = r["conversation"]

        # Try to find original stressor for baseline_markers
        stressor = find_stressor_by_id(sid, stype)

        # We need the system_prompt. Try stressor first, then extract from conversation metadata.
        system_prompt = ""
        if stressor:
            system_prompt = stressor.get("system_prompt", "")
        if not system_prompt:
            # Fallback: check if it was saved in the result
            system_prompt = r.get("system_prompt", "You are a helpful assistant.")

        try:
            score = judge.evaluate(
                conversation=conversation,
                system_prompt=system_prompt,
                target_dimensions=stressor.get("target_dimensions") if stressor else None,
                stressor=stressor,
            )
            all_scores.append(score)
            r["score"] = score.to_dict()
            r["judge_error"] = None
            if "system_prompt" not in r:
                r["system_prompt"] = system_prompt
        except Exception as e:
            print("\nJudge failed for %s: %s" % (sid, e))
            r["score"] = None
            r["judge_error"] = str(e)
            errors += 1

        rescored.append(r)

    print("\nCompleted: %d/%d | Errors: %d" % (len(all_scores), len(results), errors))

    # Save rescored results
    os.makedirs(output_dir, exist_ok=True)

    # Derive filename from input
    input_stem = Path(results_path).stem
    rescored_path = os.path.join(output_dir, "%s_rescored.json" % input_stem)
    with open(rescored_path, "w", encoding="utf-8") as f:
        json.dump(rescored, f, indent=2, ensure_ascii=False)

    if all_scores:
        report_path = os.path.join(output_dir, "%s_rescored_report.json" % input_stem)
        generate_report(
            scores=all_scores,
            model_name=model_name,
            output_path=report_path,
        )
        print("\nRescored results: %s" % rescored_path)
        print("Rescored report:  %s" % report_path)
    else:
        print("\nRescored results: %s" % rescored_path)
        print("(All judge calls failed)")

    return rescored, all_scores


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ContinuityBench Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_eval.py --model deepseek/deepseek-chat --stressors all
  python run_eval.py --model openai/gpt-4o --stressors domain_switch,style_pull
  python run_eval.py --model anthropic/claude-sonnet-4-6 --workers 5 --output results/
  python run_eval.py --model deepseek/deepseek-reasoner --stressors all --skip-judge
  python run_eval.py --rescore results/deepseek_deepseek-chat_traditional.json --judge-model openai/anthropic/claude-sonnet-4-6
        """,
    )
    parser.add_argument(
        "--model", required=False, default=None,
        help="Target model as provider/model (e.g., deepseek/deepseek-chat). Not needed for --rescore."
    )
    parser.add_argument(
        "--stressors", default="all",
        help="Comma-separated stressor types or 'all' (default: all)"
    )
    parser.add_argument(
        "--config", default="configs/default",
        help="Path to evaluation config (default: configs/default)"
    )
    parser.add_argument(
        "--output", default="results/",
        help="Output directory for results (default: results/)"
    )
    parser.add_argument(
        "--max-variants", type=int, default=None,
        help="Override max variants per stressor type"
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="Override judge model (e.g., deepseek/deepseek-chat)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load stressors and show plan without running"
    )
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Run conversations only, without judge scoring"
    )
    parser.add_argument(
        "--judge-mode", default="traditional",
        choices=["vanilla", "traditional", "sef"],
        help="Judge mode: 'vanilla' (original), 'traditional' (primary/secondary dims), 'sef' (SEF protocol with anchors + deliberation)"
    )
    parser.add_argument(
        "--energy-aware", action="store_true",
        help="(SEF mode only) Allocate judge passes based on D/A/G energy states. Reduces API calls ~40-50%% while preserving score quality."
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="Number of parallel workers (default: 5)"
    )
    parser.add_argument(
        "--rescore", default=None,
        help="Path to existing results JSON. Re-runs judge only (no new conversations). Requires --judge-model."
    )
    parser.add_argument(
        "--stressor-ids", default=None,
        help="Comma-separated stressor IDs to run (e.g. gi_v2_001,ah_v2_001). Overrides --stressors and --max-variants."
    )
    parser.add_argument(
        "--num-runs", type=int, default=1, metavar="N",
        help="Number of times to run each stressor (results are averaged). Default: 1."
    )
    parser.add_argument(
        "--model-name", default=None,
        help="Display name for the model in reports and leaderboard (e.g. 'Claude Opus 4.6'). Defaults to --model if not provided."
    )
    parser.add_argument(
        "--leaderboard", action="store_true",
        help="Run only the 26 leaderboard stressor variants (reads IDs from configs/default). Equivalent to --stressor-ids with the official set."
    )
    args = parser.parse_args()

    # ---- Rescore mode ----
    if args.rescore:
        if not os.path.exists(args.rescore):
            print("Error: Results file not found: %s" % args.rescore)
            sys.exit(1)

        # Load config
        config = {}
        if os.path.exists(args.config):
            with open(args.config, encoding="utf-8") as f:
                config = yaml.safe_load(f)

        scoring_cfg = config.get("scoring", {})
        reference_judge = scoring_cfg.get("reference_judge_model", "openai/gpt-5-mini")
        judge_model = args.judge_model or scoring_cfg.get("judge_model", reference_judge)
        judge_passes = scoring_cfg.get("judge_passes", 3)
        judge_temp = scoring_cfg.get("judge_temperature", 0.1)

        is_reference_judge = (judge_model == reference_judge)
        print("Judge model: %s%s" % (judge_model, "" if is_reference_judge else " [non-reference]"))
        if not is_reference_judge:
            print("  Reference judge is: %s" % reference_judge)
            print("  Results are non-reference and should not be compared to official leaderboard scores.")

        if args.judge_mode == "sef":
            ea = getattr(args, "energy_aware", False)
            judge = SEFJudgeSystem(SEFJudgeConfig(
                model=judge_model,
                passes=judge_passes,
                temperature=judge_temp,
                enable_deliberation=True,
                energy_aware=ea,
            ))
        else:
            judge = JudgeSystem(JudgeConfig(
                model=judge_model,
                passes=judge_passes,
                temperature=judge_temp,
            ))

        rescore_results(args.rescore, judge, args.output, model_name=args.judge_model or "rescored")
        return

    # ---- Normal mode ----
    if not args.model:
        print("Error: --model is required (unless using --rescore).")
        sys.exit(1)

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)

    # --leaderboard flag: load IDs from config
    if args.leaderboard:
        lb_ids = config.get("stressors", {}).get("leaderboard_ids", [])
        if not lb_ids:
            print("Error: No leaderboard_ids found in config file: %s" % args.config)
            sys.exit(1)
        args.stressor_ids = ",".join(lb_ids)
        args.stressors = "all"  # need all types loaded to filter by ID
        print("Leaderboard mode: %d stressor variants" % len(lb_ids))

    # Determine stressor types
    if args.stressors == "all":
        stressor_types = list(STRESSOR_FILES.keys())
    else:
        stressor_types = [s.strip() for s in args.stressors.split(",")]

    # Load stressors
    max_variants = args.max_variants or config.get("evaluation", {}).get("variants_per_stressor", 10)
    stressors = load_stressors(stressor_types, max_variants)

    # Filter by explicit stressor IDs if provided
    if args.stressor_ids:
        ids_set = {s.strip() for s in args.stressor_ids.split(",")}
        stressors = [s for s in stressors if s["id"] in ids_set]
        missing = ids_set - {s["id"] for s in stressors}
        if missing:
            print("Warning: stressor IDs not found: %s" % ", ".join(sorted(missing)))

    if not stressors:
        print("No stressors loaded. Check your stressor files.")
        sys.exit(1)

    num_runs = args.num_runs
    stressor_runs = [(s, r) for s in stressors for r in range(num_runs)]
    total_runs = len(stressor_runs)
    workers = min(args.workers, total_runs)

    display_name = args.model_name or args.model

    print("\nContinuityBench Evaluation")
    print("=" * 40)
    if args.model_name and args.model_name != args.model:
        print("Target Model:   %s (display: %s)" % (args.model, args.model_name))
    else:
        print("Target Model:   %s" % args.model)
    print("Stressor Types: %s" % ", ".join(stressor_types))
    print("Total Variants: %d" % len(stressors))
    if num_runs > 1:
        print("Runs per stressor: %d (total: %d)" % (num_runs, total_runs))
    print("Workers:        %d" % workers)
    print("Output:         %s" % args.output)
    print()

    if args.dry_run:
        print("Dry run - stressor plan:")
        for s in stressors:
            print("  [%s] %s (difficulty: %s, turns: %d, targets: %s)" % (
                s["type"], s["id"], s["difficulty"], len(s["turns"]), s["target_dimensions"]))
        print("\nTotal: %d conversations to run" % len(stressors))
        return

    # Initialize model client
    provider, model_name, client = create_model_client(args.model)

    # Initialize judge
    judge = None
    is_reference_judge = True
    if not args.skip_judge:
        scoring_cfg = config.get("scoring", {})
        reference_judge = scoring_cfg.get("reference_judge_model", "openai/gpt-5-mini")
        judge_model = args.judge_model or scoring_cfg.get("judge_model", reference_judge)
        judge_passes = scoring_cfg.get("judge_passes", 3)
        judge_temp = scoring_cfg.get("judge_temperature", 0.1)

        is_reference_judge = (judge_model == reference_judge)
        if not is_reference_judge:
            print("[!] Non-reference judge: %s (reference: %s)" % (judge_model, reference_judge))
            print("    Results should be marked 'non-reference' — do not compare directly to leaderboard scores.")

        if args.judge_mode == "sef":
            ea = getattr(args, "energy_aware", False)
            print("Judge mode: SEF Protocol (anchors + deliberation%s)" % (", energy-aware" if ea else ""))
            judge = SEFJudgeSystem(SEFJudgeConfig(
                model=judge_model,
                passes=judge_passes,
                temperature=judge_temp,
                enable_deliberation=True,
                energy_aware=ea,
            ))
        else:
            # "vanilla" and "traditional" use the same JudgeSystem;
            # the difference is that "traditional" marks primary/secondary dims
            # via target_dimensions (already wired through run_single_stressor)
            print("Judge mode: %s" % args.judge_mode)
            judge = JudgeSystem(JudgeConfig(
                model=judge_model,
                passes=judge_passes,
                temperature=judge_temp,
            ))

    # Run evaluations in parallel
    temperature = config.get("evaluation", {}).get("temperature", 0.7)
    all_scored = []   # list of (stressor_id, BCScore)
    all_conversations = []
    errors = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_single_stressor,
                provider, model_name, client, stressor, temperature, judge
            ): (stressor, run_idx)
            for stressor, run_idx in stressor_runs
        }

        with tqdm(total=total_runs, desc="Running stressors") as pbar:
            for future in as_completed(futures):
                stressor, run_idx = futures[future]
                try:
                    result, score = future.result()
                    all_conversations.append(result)
                    if score is not None:
                        all_scored.append((stressor["id"], score))
                    pbar.set_postfix({"done": stressor["id"]})
                except Exception as e:
                    print("\nError on %s: %s" % (stressor["id"], e))
                    # Save conversation even if judge failed
                    all_conversations.append({
                        "stressor_id": stressor["id"],
                        "stressor_type": stressor["type"],
                        "difficulty": stressor["difficulty"],
                        "conversation": [],
                        "score": None,
                        "judge_error": str(e),
                    })
                    errors += 1
                pbar.update(1)

    # Group scores by stressor_id for per-stressor reporting
    from collections import defaultdict
    per_stressor_scores = defaultdict(list)
    for sid, score in all_scored:
        per_stressor_scores[sid].append(score)

    all_scores = [score for _, score in all_scored]

    print("\nCompleted: %d/%d | Errors: %d" % (len(all_scores), len(all_conversations), errors))

    if not args.skip_judge and not all_scores:
        print("No evaluations completed successfully.")
        # Still save conversations so they can be rescored later
        if all_conversations:
            os.makedirs(args.output, exist_ok=True)
            safe_model_name = args.model.replace("/", "_")
            mode_suffix = "" if args.judge_mode == "vanilla" else "_%s" % args.judge_mode
            results_path = os.path.join(args.output, "%s%s.json" % (safe_model_name, mode_suffix))
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(all_conversations, f, indent=2, ensure_ascii=False)
            print("Conversations saved (no scores): %s" % results_path)
            print("You can rescore later with: python run_eval.py --rescore %s --judge-model <model>" % results_path)
        sys.exit(1)

    # Generate output
    os.makedirs(args.output, exist_ok=True)
    safe_model_name = args.model.replace("/", "_")

    # Add judge mode suffix to avoid overwriting across modes
    mode_suffix = "" if args.judge_mode == "vanilla" else "_%s" % args.judge_mode
    if getattr(args, "energy_aware", False):
        mode_suffix += "_energy"

    # Save full results (UTF-8 for Windows compatibility)
    results_path = os.path.join(args.output, "%s%s.json" % (safe_model_name, mode_suffix))
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_conversations, f, indent=2, ensure_ascii=False)

    # Generate and display report
    if all_scores:
        report_path = os.path.join(args.output, "%s%s_report.json" % (safe_model_name, mode_suffix))
        judge_metadata = {
            "judge_model": judge_model if not args.skip_judge else None,
            "is_reference_judge": is_reference_judge,
            "judge_mode": args.judge_mode,
        }
        generate_report(
            scores=all_scores,
            model_name=display_name,
            output_path=report_path,
            per_stressor=dict(per_stressor_scores),
            num_runs=num_runs,
            metadata=judge_metadata,
        )
        print("\nFull results: %s" % results_path)
        print("Report:       %s" % report_path)
    else:
        print("\nFull results: %s" % results_path)
        print("(Judge scoring was skipped)")


if __name__ == "__main__":
    main()
