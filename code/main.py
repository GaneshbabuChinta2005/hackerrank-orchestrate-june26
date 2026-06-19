"""
main.py — Entry point for the HackerRank Orchestrate claim verification system.

Usage:
    python main.py [--input dataset/claims.csv] [--output output.csv] [--no-cache]

Reads dataset/claims.csv, processes each claim using Gemini Vision,
and writes output.csv to the repository root.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure code/ directory is on sys.path regardless of working directory
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CODE_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from analyzer import analyze_claim
from config import CACHE_FILE, OUTPUT_COLUMNS
from risk import apply_history_risk
from utils import (
    load_csv,
    load_evidence_requirements,
    load_user_history,
    make_cache_key,
    load_cache,
    save_cache,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Damage claim verification pipeline")
    parser.add_argument(
        "--input",
        default=os.path.join(REPO_ROOT, "dataset", "claims.csv"),
        help="Path to input claims CSV (default: dataset/claims.csv)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "output.csv"),
        help="Path to write output CSV (default: output.csv in repo root)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable result caching (re-process all rows)",
    )
    return parser.parse_args()


def run_pipeline(
    claims_path: str,
    output_path: str,
    use_cache: bool = True,
) -> list[dict]:
    """
    Main pipeline: reads claims, calls Gemini for each, applies risk overlay, writes output.
    Returns the list of output rows.
    """
    dataset_root = os.path.join(REPO_ROOT, "dataset")
    history_path = os.path.join(dataset_root, "user_history.csv")
    reqs_path = os.path.join(dataset_root, "evidence_requirements.csv")
    cache_path = os.path.join(CODE_DIR, CACHE_FILE)

    print(f"Loading claims from: {claims_path}")
    claims = load_csv(claims_path)
    print(f"Loaded {len(claims)} claims.")

    print(f"Loading user history from: {history_path}")
    user_history_map = load_user_history(history_path)

    print(f"Loading evidence requirements from: {reqs_path}")
    evidence_requirements = load_evidence_requirements(reqs_path)

    # Load cache
    cache = load_cache(cache_path) if use_cache else {}
    print(f"Cache has {len(cache)} entries. {'(caching enabled)' if use_cache else '(caching disabled)'}")

    output_rows = []
    total = len(claims)

    for i, claim in enumerate(claims, start=1):
        user_id = claim.get("user_id", "unknown")
        claim_object = claim.get("claim_object", "unknown")
        image_paths = claim.get("image_paths", "")
        print(f"\n[{i}/{total}] Processing claim for {user_id} | object={claim_object} | images={image_paths}")

        cache_key = make_cache_key(claim)

        if use_cache and cache_key in cache:
            cached = cache[cache_key]
            # Skip stale fallback results from API errors — re-process them
            justification = cached.get("claim_status_justification", "")
            if "API error" in justification or "Analysis failed" in justification:
                print(f"  [CACHE STALE] Cached result was a fallback — re-processing...")
            else:
                print(f"  [CACHE HIT] Using cached result.")
                analysis = cached
                # Build output row (in required column order)
                row = {
                    "user_id": user_id,
                    "image_paths": image_paths,
                    "user_claim": claim.get("user_claim", ""),
                    "claim_object": claim_object,
                    **analysis,
                }
                for col in OUTPUT_COLUMNS:
                    if col not in row:
                        row[col] = ""
                output_rows.append(row)
                continue

        user_history = user_history_map.get(user_id, {})
        analysis = analyze_claim(claim, user_history, evidence_requirements, dataset_root)
        # Apply user history risk flags post-inference
        analysis = apply_history_risk(analysis, user_history)
        if use_cache:
            cache[cache_key] = analysis
            save_cache(cache_path, cache)

        # Build output row (in required column order)
        row = {
            "user_id": user_id,
            "image_paths": image_paths,
            "user_claim": claim.get("user_claim", ""),
            "claim_object": claim_object,
            **analysis,
        }

        # Ensure all output columns exist
        for col in OUTPUT_COLUMNS:
            if col not in row:
                row[col] = ""

        output_rows.append(row)

    print(f"\nWriting {len(output_rows)} rows to: {output_path}")
    write_csv(output_path, output_rows)
    print("Done! output.csv written successfully.")
    return output_rows


def main():
    args = parse_args()
    start = time.time()
    run_pipeline(
        claims_path=args.input,
        output_path=args.output,
        use_cache=not args.no_cache,
    )
    elapsed = time.time() - start
    print(f"\nTotal runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
