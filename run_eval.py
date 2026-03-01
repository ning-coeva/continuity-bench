#!/usr/bin/env python3
"""
ContinuityBench Evaluation Runner

Usage:
    python run_eval.py --model openai/gpt-5.2 --stressors all
    python run_eval.py --model anthropic/claude-sonnet-4-6 --stressors domain_switch
    python run_eval.py --model openai/gpt-5.2 --config configs/default.yaml --output results/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import jsonlines
import yaml
from tqdm import tqdm

from scoring.bc_score import BCScore
from scoring.judges import JudgeSystem, JudgeConfig
from scoring.report import generate_report


# ============================================================
# Model Interaction
# ============================================================

def create_model_client(model_spec: str):
    """Create an API client for the target model."""
    provider, model_name = model_spec.split("/", 1)

    if provider == "openai":
        import openai
        return "openai", model_name, openai.OpenAI()
    elif provider == "anthropic":
        import anthropic
        return "anthropic", model_name, anthropic.Anthropic()
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_model_response(
    provider: str,
    model_name: str,
    client,
    system_prompt: str,
    messages: list[dict],
    temperature: float = 0.7,
) -> str:
    """Get a response from the target model."""
    if provider == "openai":
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                *messages,
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content

    elif provider == "anthropic":
        response = client.messages.create(
            model=model_name,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            temperature=temperature,
        )
        return response.content[0].text


# ============================================================
# Stressor Loading
# ============================================================

STRESSOR_FILES = {
    "domain_switch": "stressors/domain_switch.jsonl",
    "abstraction_hop": "stressors/abstraction_hop.jsonl",
    "goal_interrupt": "stressors/goal_interrupt.jsonl",
    "style_pull": "stressors/style_pull.jsonl",
}


def load_stressors(stressor_types: list[str], max_variants: int = 30) -> list[dict]:
    """Load stressor sequences from JSONL files."""
    stressors = []
    for st in stressor_types:
        if st not in STRESSOR_FILES:
            print(f"Warning: Unknown stressor type '{st}', skipping.")
            continue
        filepath = STRESSOR_FILES[st]
        if not os.path.exists(filepath):
            print(f"Warning: Stressor file not found: {filepath}")
            continue
        with jsonlines.open(filepath) as reader:
            for i, item in enumerate(reader):
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
) -> list[dict]:
    """
    Run a stressor conversation against the target model.

    Returns the full conversation (user + assistant turns).
    """
    system_prompt = stressor["system_prompt"]
    conversation = []

    for turn in stressor["turns"]:
        # Add user turn
        conversation.append({
            "role": "user",
            "content": turn["content"],
            "phase": turn.get("phase", "unknown"),
            "turn_id": turn.get("turn_id", len(conversation) + 1),
        })

        # Get model response
        # Only pass role + content to the API (strip metadata)
        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in conversation
        ]
        response = get_model_response(
            provider, model_name, client,
            system_prompt, api_messages, temperature,
        )

        # Add assistant turn
        conversation.append({
            "role": "assistant",
            "content": response,
            "turn_id": turn.get("turn_id", len(conversation)),
        })

    return conversation


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ContinuityBench Evaluation Runner")
    parser.add_argument(
        "--model", required=True,
        help="Target model (e.g., openai/gpt-5.2, anthropic/claude-sonnet-4-6)"
    )
    parser.add_argument(
        "--stressors", default="all",
        help="Comma-separated stressor types or 'all'"
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to evaluation config"
    )
    parser.add_argument(
        "--output", default="results/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-variants", type=int, default=None,
        help="Override max variants per stressor type"
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="Override judge model (e.g., openai/gpt-5.2)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load stressors and show plan without running"
    )
    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Determine stressor types
    if args.stressors == "all":
        stressor_types = list(STRESSOR_FILES.keys())
    else:
        stressor_types = [s.strip() for s in args.stressors.split(",")]

    # Load stressors
    max_variants = args.max_variants or config.get("evaluation", {}).get("variants_per_stressor", 10)
    stressors = load_stressors(stressor_types, max_variants)

    if not stressors:
        print("No stressors loaded. Check your stressor files.")
        sys.exit(1)

    print(f"\nContinuityBench Evaluation")
    print(f"{'='*40}")
    print(f"Target Model:   {args.model}")
    print(f"Stressor Types: {', '.join(stressor_types)}")
    print(f"Total Variants: {len(stressors)}")
    print(f"Output:         {args.output}")
    print()

    if args.dry_run:
        print("Dry run — stressor plan:")
        for s in stressors:
            print(f"  [{s['type']}] {s['id']} (difficulty: {s['difficulty']}, "
                  f"turns: {len(s['turns'])}, targets: {s['target_dimensions']})")
        print(f"\nTotal: {len(stressors)} conversations to run")
        return

    # Initialize model client
    provider, model_name, client = create_model_client(args.model)

    # Initialize judge
    judge_model = args.judge_model or config.get("scoring", {}).get("judge_model", "openai/gpt-5.2")
    judge_passes = config.get("scoring", {}).get("judge_passes", 3)
    judge_temp = config.get("scoring", {}).get("judge_temperature", 0.1)
    judge = JudgeSystem(JudgeConfig(
        model=judge_model,
        passes=judge_passes,
        temperature=judge_temp,
    ))

    # Run evaluations
    temperature = config.get("evaluation", {}).get("temperature", 0.7)
    all_scores = []
    all_conversations = []

    for stressor in tqdm(stressors, desc="Running stressors"):
        try:
            # Run conversation
            conversation = run_conversation(
                provider, model_name, client, stressor, temperature
            )

            # Judge the conversation
            score = judge.evaluate(
                conversation=conversation,
                system_prompt=stressor["system_prompt"],
                target_dimensions=stressor.get("target_dimensions"),
            )

            all_scores.append(score)
            all_conversations.append({
                "stressor_id": stressor["id"],
                "stressor_type": stressor["type"],
                "difficulty": stressor["difficulty"],
                "conversation": conversation,
                "score": score.to_dict(),
            })

        except Exception as e:
            print(f"\nError on {stressor['id']}: {e}")
            continue

    if not all_scores:
        print("No evaluations completed successfully.")
        sys.exit(1)

    # Generate output
    os.makedirs(args.output, exist_ok=True)
    safe_model_name = args.model.replace("/", "_")

    # Save full results
    results_path = os.path.join(args.output, f"{safe_model_name}.json")
    with open(results_path, "w") as f:
        json.dump(all_conversations, f, indent=2, ensure_ascii=False)

    # Generate and display report
    report_path = os.path.join(args.output, f"{safe_model_name}_report.json")
    generate_report(
        scores=all_scores,
        model_name=args.model,
        output_path=report_path,
    )

    print(f"\nFull results: {results_path}")
    print(f"Report:       {report_path}")


if __name__ == "__main__":
    main()
