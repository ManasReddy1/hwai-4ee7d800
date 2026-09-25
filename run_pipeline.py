#!/usr/bin/env python3
"""Main pipeline: Compact Context + Evals + RAG for all patients.

This is the main entry point for the entire project. It reads raw FHIR
patient bundles (big JSON files), compresses them into small markdown
summaries, builds a lookup index for labs, and runs quality checks (evals).

Usage:
    python run_pipeline.py                    # Process all patients
    python run_pipeline.py Merlene            # Process only Merlene
    python run_pipeline.py --eval-only        # Run evals on existing output
"""

# --- Standard library imports ---
import json       # For reading/writing JSON files
import sys        # For reading command-line arguments (e.g., "python run_pipeline.py Merlene")
import os         # For checking environment variables (like API keys)
from pathlib import Path  # For easy file path handling (cross-platform)

# --- Our custom modules (in src/ folder) ---
from src.fhir_parser import (
    extract_observations,  # Pulls out lab results (blood tests, vitals, etc.)
    load_bundle,           # Reads a raw FHIR JSON file from disk
    parse_bundle,          # Organizes the raw data by resource type (Patient, Condition, etc.)
)
from src.compactor import compact_bundle    # Compresses parsed data into a short markdown summary
from src.rag import PatientIndex            # Builds a searchable index of lab results by code/name
from src.evals import run_generic_evals, run_merlene_evals  # Quality checks for the output

# --- Configuration ---
DATA_DIR = Path("data")      # Where the raw FHIR JSON files live
OUTPUT_DIR = Path("output")  # Where we save our compressed output


def estimate_tokens(text: str) -> int:
    """Estimate how many tokens a piece of text will use in an LLM's context window.

    LLMs don't count words — they count "tokens" (roughly 4 characters = 1 token).
    If we have an Anthropic API key, we use their exact counter. Otherwise, we
    approximate with chars/4, which is close enough for planning purposes.
    """
    # Try the accurate method first (requires an API key)
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            from anthropic import Anthropic

            result = Anthropic().messages.count_tokens(
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": text}],
            )
            return result.input_tokens
    except Exception:
        pass  # If anything goes wrong, fall back to the estimate

    # Fallback: divide character count by 4 (rough but standard approximation)
    return len(text) // 4


def get_patient_files(filter_name: str | None = None) -> list[Path]:
    """Find all patient JSON files in the data directory.

    Skips non-patient files (hospitals, practitioners) since those aren't
    individual patient records. Optionally filters by name so you can
    process just one patient (e.g., "Merlene") for debugging.
    """
    files = []
    for f in sorted(DATA_DIR.glob("*.json")):  # Get all JSON files, sorted alphabetically
        # Skip hospital and practitioner files — they aren't patient bundles
        if f.name.startswith("hospital") or f.name.startswith("practitioner"):
            continue
        # If a filter was given (e.g., "Merlene"), skip files that don't match
        if filter_name and filter_name.lower() not in f.name.lower():
            continue
        files.append(f)
    return files


def process_patient(path: Path) -> dict:
    """Process a single patient file through the full pipeline.

    This is the core function. For one patient, it:
    1. Loads the raw FHIR bundle (a big JSON blob)
    2. Parses it into organized resource types
    3. Compacts it into a short markdown summary
    4. Builds a RAG index for lab lookups
    5. Runs quality evals to verify nothing was lost
    6. Saves the compact context and RAG index to disk
    """
    print(f"  Processing: {path.name}...")

    # Step 1: Load the raw JSON file from disk
    bundle = load_bundle(path)

    # Step 2: Parse into organized groups (Patient, Condition, Observation, etc.)
    parsed = parse_bundle(bundle)
    resources = parsed["resources"]  # Dict like {"Patient": [...], "Condition": [...], ...}

    # Step 3: Compress everything into a short markdown summary
    context = compact_bundle(resources)
    raw_size = path.stat().st_size         # Original file size in bytes
    compact_tokens = estimate_tokens(context)  # How many tokens the summary uses

    # Step 4: Build a searchable index so we can look up specific labs by code or name
    index = PatientIndex(resources)
    available_labs = index.lookup_all_lab_codes()  # List of all lab types available

    # Step 5: Run quality checks (evals)
    # Merlene gets special evals (6 questions), everyone else gets generic ones (4 questions)
    is_merlene = "merlene" in path.name.lower()
    observations = extract_observations(resources)

    if is_merlene:
        eval_suite = run_merlene_evals(context, observations)
    else:
        eval_suite = run_generic_evals(context, resources)

    # Build a result dictionary with all the metrics we want to track
    result = {
        "file": path.name,
        # Extract patient name from the first line of compact context ("# Patient: Name")
        "patient_name": context.split("\n")[0].replace("# Patient: ", ""),
        "raw_bytes": raw_size,
        "compact_chars": len(context),
        "compact_tokens": compact_tokens,
        "token_method": "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "chars/4",
        # How much we compressed: e.g., "99.2%" means we removed 99.2% of the data
        "compression_ratio": f"{(1 - compact_tokens / (raw_size // 4)) * 100:.1f}%",
        # The whole point: is the compact version under 900K tokens?
        "under_900k": compact_tokens <= 900_000,
        "num_observations": sum(len(v) for v in observations.values()),
        "distinct_obs_types": len(observations),
        "available_labs": len(available_labs),
        "eval_total": eval_suite.total,
        "eval_passed": eval_suite.passed,
        "eval_pass_rate": f"{eval_suite.pass_rate:.0%}",
        # Store each eval question/answer for detailed inspection later
        "eval_details": [
            {
                "question": r.question,
                "expected": r.expected,
                "actual": r.actual,
                "passed": r.passed,
                "category": r.category,
            }
            for r in eval_suite.results
        ],
    }

    # Step 6a: Save the compact markdown summary to disk
    output_path = OUTPUT_DIR / path.name.replace(".json", "_compact.md")
    output_path.write_text(context)

    # Step 6b: Save the RAG lab index (list of available lab codes) to disk
    rag_path = OUTPUT_DIR / path.name.replace(".json", "_rag.json")
    rag_path.write_text(json.dumps(available_labs, indent=2, default=str))

    return result


def main():
    """Entry point: process all patients and print a summary."""

    # Create output directory if it doesn't exist yet
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Check if user wants to process only one patient (e.g., "python run_pipeline.py Merlene")
    filter_name = None
    if len(sys.argv) > 1 and sys.argv[1] != "--eval-only":
        filter_name = sys.argv[1]

    # Find all matching patient files
    patient_files = get_patient_files(filter_name)
    print(f"Found {len(patient_files)} patient files to process.\n")

    # Process each patient one by one, collecting results
    all_results = []
    for pf in patient_files:
        result = process_patient(pf)
        all_results.append(result)

        # Print a one-line summary for each patient as we go
        print(f"    {result['patient_name']}: "
              f"{result['raw_bytes']:,} bytes -> {result['compact_tokens']:,} tokens "
              f"({result['compression_ratio']} reduction) | "
              f"Evals: {result['eval_passed']}/{result['eval_total']}")

    # Save all results to a single JSON file (the dashboard reads this)
    results_path = OUTPUT_DIR / "pipeline_results.json"
    results_path.write_text(json.dumps(all_results, indent=2, default=str))

    # --- Print final summary ---
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Patients processed: {len(all_results)}")
    print(f"All under 900K tokens: {all(r['under_900k'] for r in all_results)}")
    print(f"Token counting method: {all_results[0]['token_method'] if all_results else 'N/A'}")

    # Calculate overall eval pass rate across all patients
    total_evals = sum(r["eval_total"] for r in all_results)
    total_passed = sum(r["eval_passed"] for r in all_results)
    print(f"Total evals: {total_passed}/{total_evals} passed "
          f"({total_passed/total_evals:.0%})" if total_evals else "")

    # If Merlene was processed, show her detailed eval results (she's the reference patient)
    merlene = [r for r in all_results if "merlene" in r["file"].lower()]
    if merlene:
        m = merlene[0]
        print(f"\nMerlene eval details:")
        for e in m["eval_details"]:
            status = "PASS" if e["passed"] else "FAIL"
            print(f"  [{status}] {e['question']}")
            if not e["passed"]:
                print(f"         Expected: {e['expected']}")
                print(f"         Got: {e['actual']}")

    # Tell the user where to find the output files
    print(f"\nOutput saved to: {OUTPUT_DIR}/")
    print(f"  - *_compact.md   : Compact contexts")
    print(f"  - *_rag.json     : RAG lab indices")
    print(f"  - pipeline_results.json : Full results")


# This is the standard Python entry point — runs main() when you execute the script directly
if __name__ == "__main__":
    main()
