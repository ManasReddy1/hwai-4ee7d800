#!/usr/bin/env python3
"""Main pipeline: Compact Context + Evals + RAG for all patients.

Usage:
    python run_pipeline.py                    # Process all patients
    python run_pipeline.py Merlene            # Process only Merlene
    python run_pipeline.py --eval-only        # Run evals on existing output
"""

import json
import sys
import os
from pathlib import Path

from src.fhir_parser import (
    extract_observations,
    load_bundle,
    parse_bundle,
)
from src.compactor import compact_bundle
from src.rag import PatientIndex
from src.evals import run_generic_evals, run_merlene_evals

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


def estimate_tokens(text: str) -> int:
    """Estimate token count. Uses Anthropic API if available, else chars/4."""
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            from anthropic import Anthropic

            result = Anthropic().messages.count_tokens(
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": text}],
            )
            return result.input_tokens
    except Exception:
        pass

    return len(text) // 4


def get_patient_files(filter_name: str | None = None) -> list[Path]:
    """Get patient bundle files, optionally filtered by name."""
    files = []
    for f in sorted(DATA_DIR.glob("*.json")):
        if f.name.startswith("hospital") or f.name.startswith("practitioner"):
            continue
        if filter_name and filter_name.lower() not in f.name.lower():
            continue
        files.append(f)
    return files


def process_patient(path: Path) -> dict:
    """Process a single patient: compact, index, eval."""
    print(f"  Processing: {path.name}...")

    bundle = load_bundle(path)
    parsed = parse_bundle(bundle)
    resources = parsed["resources"]

    # Compact
    context = compact_bundle(resources)
    raw_size = path.stat().st_size
    compact_tokens = estimate_tokens(context)

    # Build RAG index
    index = PatientIndex(resources)
    available_labs = index.lookup_all_lab_codes()

    # Run evals
    is_merlene = "merlene" in path.name.lower()
    observations = extract_observations(resources)

    if is_merlene:
        eval_suite = run_merlene_evals(context, observations)
    else:
        eval_suite = run_generic_evals(context, resources)

    result = {
        "file": path.name,
        "patient_name": context.split("\n")[0].replace("# Patient: ", ""),
        "raw_bytes": raw_size,
        "compact_chars": len(context),
        "compact_tokens": compact_tokens,
        "token_method": "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "chars/4",
        "compression_ratio": f"{(1 - compact_tokens / (raw_size // 4)) * 100:.1f}%",
        "under_900k": compact_tokens <= 900_000,
        "num_observations": sum(len(v) for v in observations.values()),
        "distinct_obs_types": len(observations),
        "available_labs": len(available_labs),
        "eval_total": eval_suite.total,
        "eval_passed": eval_suite.passed,
        "eval_pass_rate": f"{eval_suite.pass_rate:.0%}",
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

    # Save compact context
    output_path = OUTPUT_DIR / path.name.replace(".json", "_compact.md")
    output_path.write_text(context)

    # Save RAG index summary
    rag_path = OUTPUT_DIR / path.name.replace(".json", "_rag.json")
    rag_path.write_text(json.dumps(available_labs, indent=2, default=str))

    return result


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    filter_name = None
    if len(sys.argv) > 1 and sys.argv[1] != "--eval-only":
        filter_name = sys.argv[1]

    patient_files = get_patient_files(filter_name)
    print(f"Found {len(patient_files)} patient files to process.\n")

    all_results = []
    for pf in patient_files:
        result = process_patient(pf)
        all_results.append(result)

        # Print summary for each patient
        print(f"    {result['patient_name']}: "
              f"{result['raw_bytes']:,} bytes -> {result['compact_tokens']:,} tokens "
              f"({result['compression_ratio']} reduction) | "
              f"Evals: {result['eval_passed']}/{result['eval_total']}")

    # Save full results
    results_path = OUTPUT_DIR / "pipeline_results.json"
    results_path.write_text(json.dumps(all_results, indent=2, default=str))

    # Print summary
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Patients processed: {len(all_results)}")
    print(f"All under 900K tokens: {all(r['under_900k'] for r in all_results)}")
    print(f"Token counting method: {all_results[0]['token_method'] if all_results else 'N/A'}")

    total_evals = sum(r["eval_total"] for r in all_results)
    total_passed = sum(r["eval_passed"] for r in all_results)
    print(f"Total evals: {total_passed}/{total_evals} passed "
          f"({total_passed/total_evals:.0%})" if total_evals else "")

    # Show Merlene details if processed
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

    print(f"\nOutput saved to: {OUTPUT_DIR}/")
    print(f"  - *_compact.md   : Compact contexts")
    print(f"  - *_rag.json     : RAG lab indices")
    print(f"  - pipeline_results.json : Full results")


if __name__ == "__main__":
    main()
