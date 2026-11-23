#!/usr/bin/env python3
"""
CLI Tool for Groq Prompt Shot Classification & Testing
Allows analyzing and checking whether any prompt is zero-shot, one-shot, or few-shot directly in the terminal.
"""

import sys
import os
import argparse
import json
from detector import PromptShotDetector
from groq_service import GroqService


def print_banner():
    print("=" * 65)
    print(" ⚡ GROQ LLM PLAYGROUND: PROMPT SHOT DETECTOR & ANALYZER ⚡ ")
    print("=" * 65)


def print_heuristic_result(result):
    classification = result["classification"].upper()
    shot_count = result["shot_count"]
    confidence = int(result["confidence"] * 100)

    color_prefix = ""
    color_reset = ""
    # ANSI color codes
    if classification == "ZERO-SHOT":
        color_prefix = "\033[94m" # Blue
    elif classification == "ONE-SHOT":
        color_prefix = "\033[93m" # Yellow
    elif classification == "FEW-SHOT":
        color_prefix = "\033[92m" # Green
    color_reset = "\033[0m"

    print(f"\n📊 CLASSIFICATION: {color_prefix}{classification}{color_reset}")
    print(f"🔢 Solved Examples Detected: {shot_count}")
    print(f"🎯 Confidence: {confidence}%")
    print(f"💡 Analysis: {result['reasoning']}")

    if result.get("patterns_found"):
        print(f"🔍 Pattern Signals: {', '.join(result['patterns_found'])}")

    if result.get("examples"):
        print("\n--- Extracted Demonstration Examples ---")
        for ex in result["examples"]:
            ex_id = ex.get("id", 1)
            inp = ex.get("input", "")
            out = ex.get("output", "")
            print(f"  [Example {ex_id}]")
            if inp:
                print(f"    • Input:  {inp}")
            if out:
                print(f"    • Output: {out}")
            if not inp and not out:
                print(f"    • Content:\n      {ex.get('content')}")

    if result.get("target_query"):
        print(f"\n🎯 Target Query / Final Prompt Task:")
        print(f"    {result['target_query']}")

    metrics = result.get("prompt_metrics", {})
    print(f"\n📏 Metrics: {metrics.get('word_count', 0)} words | {metrics.get('char_count', 0)} characters | {metrics.get('line_count', 0)} lines")


def print_deep_result(deep):
    print("\n" + "=" * 65)
    print(" 🧠 GROQ DEEP LLM SEMANTIC ANALYSIS")
    print("=" * 65)
    print(f"Classification:   {deep.get('classification', '').upper()}")
    print(f"Examples Found:   {deep.get('example_count', 0)}")
    print(f"Quality Score:    {deep.get('prompt_quality_score', 'N/A')}/100")
    print(f"Explanation:      {deep.get('explanation', '')}")

    if deep.get("techniques_detected"):
        print(f"Techniques:       {', '.join(deep['techniques_detected'])}")

    if deep.get("strengths"):
        print("\nStrengths:")
        for s in deep["strengths"]:
            print(f"  ✓ {s}")

    if deep.get("improvement_suggestions"):
        print("\nImprovement Suggestions:")
        for sug in deep["improvement_suggestions"]:
            print(f"  ➜ {sug}")

    meta = deep.get("meta", {})
    if meta:
        print(f"\nModel: {meta.get('model_used')} | Latency: {meta.get('latency_ms')}ms | Speed: {meta.get('tokens_per_second')} tok/s")


def main():
    parser = argparse.ArgumentParser(
        description="Check whether a prompt is zero-shot, one-shot, or few-shot, and test it on Groq."
    )
    parser.add_argument("prompt", nargs="?", default="", help="Prompt text to analyze")
    parser.add_argument("--file", "-f", help="Read prompt from a file")
    parser.add_argument("--deep", "-d", action="store_true", help="Run deep semantic analysis with Groq LLM")
    parser.add_argument("--run", "-r", action="store_true", help="Execute the prompt on Groq LLM and view output")
    parser.add_argument("--model", "-m", default="llama-3.3-70b-versatile", help="Groq model ID")
    parser.add_argument("--key", "-k", default="", help="Groq API key (defaults to GROQ_API_KEY env var)")
    parser.add_argument("--json", "-j", action="store_true", help="Output results as raw JSON")

    args = parser.parse_args()

    prompt_text = args.prompt
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                prompt_text = f.read()
        except Exception as e:
            print(f"Error reading file '{args.file}': {e}", file=sys.stderr)
            sys.exit(1)

    if not prompt_text:
        # If no argument, read from stdin if piped or interactive prompt
        if not sys.stdin.isatty():
            prompt_text = sys.stdin.read()
        else:
            print_banner()
            print("Enter or paste your prompt below (press Ctrl+D or Ctrl+Z on new line when done):")
            prompt_text = sys.stdin.read()

    if not prompt_text.strip():
        print("Error: No prompt provided to analyze.", file=sys.stderr)
        sys.exit(1)

    # 1. Run Heuristic Analysis
    heuristic_res = PromptShotDetector.analyze(prompt_text)

    api_key = args.key or os.getenv("GROQ_API_KEY", "")
    deep_res = None
    exec_res = None

    if args.deep:
        if not api_key:
            print("Warning: GROQ_API_KEY not set. Cannot run deep LLM analysis. Showing heuristic results only.", file=sys.stderr)
        else:
            try:
                service = GroqService(api_key=api_key)
                deep_res = service.analyze_prompt_deep(prompt_text, api_key=api_key, model=args.model)
            except Exception as e:
                print(f"Error during Groq deep analysis: {e}", file=sys.stderr)

    if args.run:
        if not api_key:
            print("Error: GROQ_API_KEY required to execute prompt on Groq.", file=sys.stderr)
            sys.exit(1)
        try:
            service = GroqService(api_key=api_key)
            exec_res = service.execute_prompt(prompt_text, model=args.model, api_key=api_key)
        except Exception as e:
            print(f"Error executing on Groq: {e}", file=sys.stderr)
            sys.exit(1)

    if args.json:
        output_payload = {
            "heuristic": heuristic_res,
            "deep": deep_res,
            "execution": exec_res
        }
        print(json.dumps(output_payload, indent=2))
        return

    print_banner()
    print_heuristic_result(heuristic_res)

    if deep_res:
        print_deep_result(deep_res)

    if exec_res:
        print("\n" + "=" * 65)
        print(f" 🚀 GROQ EXECUTION RESULT ({exec_res.get('model')})")
        print("=" * 65)
        print(exec_res.get("output", ""))
        metrics = exec_res.get("metrics", {})
        print("\n" + "-" * 65)
        print(f"Speed: {metrics.get('tokens_per_second')} tok/s | Latency: {metrics.get('latency_ms')} ms | Output Tokens: {metrics.get('completion_tokens')}")


if __name__ == "__main__":
    main()
