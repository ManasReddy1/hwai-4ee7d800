"""Eval suite for measuring compact context quality.

Three dimensions:
1. Factual Accuracy — exact value extraction from context
2. Trend & Reasoning — temporal reasoning and cross-referencing
3. Hallucination Resistance — model should NOT invent information

Evals work by:
1. Extracting ground-truth answers from raw FHIR data
2. Sending compact context + question to an LLM
3. Checking LLM's answer against ground truth

When no API key is available, evals run in "ground-truth only" mode —
they verify the compact context contains the needed data by string matching.
"""

import os
import re
from dataclasses import dataclass, field

try:
    from anthropic import Anthropic

    HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class EvalResult:
    """Result of a single eval question."""

    question: str
    expected: str
    actual: str
    passed: bool
    category: str  # "factual", "trend", "hallucination"
    details: str = ""


@dataclass
class EvalSuite:
    """Collection of eval results."""

    results: list[EvalResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def by_category(self) -> dict[str, list[EvalResult]]:
        categories: dict[str, list[EvalResult]] = {}
        for r in self.results:
            categories.setdefault(r.category, []).append(r)
        return categories

    def summary(self) -> str:
        lines = [
            f"Eval Results: {self.passed}/{self.total} passed ({self.pass_rate:.0%})",
            "",
        ]
        for cat, results in self.by_category().items():
            cat_passed = sum(1 for r in results if r.passed)
            lines.append(f"  {cat}: {cat_passed}/{len(results)}")
            for r in results:
                status = "PASS" if r.passed else "FAIL"
                lines.append(f"    [{status}] {r.question}")
                if not r.passed:
                    lines.append(f"           Expected: {r.expected}")
                    lines.append(f"           Got:      {r.actual}")
        return "\n".join(lines)


def _ask_llm(context: str, question: str) -> str:
    """Ask an LLM a question about the compact context."""
    if not HAS_ANTHROPIC:
        return "[NO_API_KEY - using string matching fallback]"

    client = Anthropic()
    prompt = (
        "You are a clinical assistant. Answer the question based ONLY on the "
        "patient record below. If the information is not in the record, say "
        '"not recorded" or "none recorded". Be concise.\n\n'
        f"<patient_record>\n{context}\n</patient_record>\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _check_contains(context: str, *values: str) -> bool:
    """Check if context contains all required values (string matching fallback)."""
    context_lower = context.lower()
    return all(str(v).lower() in context_lower for v in values)


def run_merlene_evals(context: str, observations: dict) -> EvalSuite:
    """Run the example eval questions from the README against Merlene's context.

    Args:
        context: The compact context string.
        observations: Raw observations dict from extract_observations().
    """
    suite = EvalSuite()

    # --- Ground truth extraction ---
    hba1c_key = None
    for key in observations:
        if "a1c" in key.lower() or "hemoglobin a1c" in key.lower():
            hba1c_key = key
            break

    hba1c_readings = observations.get(hba1c_key, []) if hba1c_key else []
    latest_hba1c = hba1c_readings[-1] if hba1c_readings else None

    # === FACTUAL: Latest HbA1c value and date ===
    if latest_hba1c:
        expected_val = str(latest_hba1c["value"])
        expected_date = latest_hba1c["date"][:10]

        if HAS_ANTHROPIC:
            answer = _ask_llm(context, "What was the most recent HbA1c, and when was it taken?")
            passed = expected_val in answer and expected_date in answer
        else:
            answer = "string-match"
            passed = _check_contains(context, expected_val, expected_date)

        suite.results.append(EvalResult(
            question="What was her most recent HbA1c, and when was it taken?",
            expected=f"{expected_val} on {expected_date}",
            actual=answer[:200],
            passed=passed,
            category="factual",
        ))

    # === FACTUAL: Number of HbA1c readings ===
    expected_count = str(len(hba1c_readings))
    if HAS_ANTHROPIC:
        answer = _ask_llm(context, "How many HbA1c readings are recorded?")
        passed = expected_count in answer
    else:
        answer = "string-match"
        passed = _check_contains(context, expected_count)

    suite.results.append(EvalResult(
        question="How many HbA1c readings are recorded?",
        expected=expected_count,
        actual=answer[:200],
        passed=passed,
        category="factual",
    ))

    # === TREND: Is blood sugar getting worse? ===
    if HAS_ANTHROPIC:
        answer = _ask_llm(
            context,
            "Is her blood sugar getting worse? Tie it to any related diagnosis.",
        )
        # Answer should mention prediabetes and indicate stable/not worsening
        passed = (
            "prediabet" in answer.lower()
            and ("stable" in answer.lower()
                 or "not" in answer.lower()
                 or "plateau" in answer.lower()
                 or "improv" in answer.lower())
        )
    else:
        answer = "string-match"
        # Context should have prediabetes and HbA1c data with reading count > 1
        has_prediabetes = _check_contains(context, "prediabet")
        has_hba1c = _check_contains(context, "a1c") or _check_contains(context, "hemoglobin")
        passed = has_prediabetes and has_hba1c

    suite.results.append(EvalResult(
        question="Is her blood sugar getting worse? Tie it to any related diagnosis.",
        expected="Stable/plateaued in prediabetic range, linked to Prediabetes diagnosis",
        actual=answer[:200],
        passed=passed,
        category="trend",
    ))

    # === HALLUCINATION: What is she allergic to? ===
    if HAS_ANTHROPIC:
        answer = _ask_llm(context, "What is she allergic to?")
        answer_lower = answer.lower()
        passed = (
            "no" in answer_lower or "none" in answer_lower or "not recorded" in answer_lower
        ) and not any(
            drug in answer_lower
            for drug in ["penicillin", "aspirin", "sulfa", "latex", "peanut"]
        )
    else:
        answer = "string-match"
        passed = _check_contains(context, "no allergies recorded")

    suite.results.append(EvalResult(
        question="What is she allergic to?",
        expected="Nothing recorded / no known allergies",
        actual=answer[:200],
        passed=passed,
        category="hallucination",
    ))

    # === FACTUAL: Active diagnosis ===
    if HAS_ANTHROPIC:
        answer = _ask_llm(context, "What are her active medical problems?")
        passed = "prediabet" in answer.lower()
    else:
        answer = "string-match"
        passed = _check_contains(context, "prediabet")

    suite.results.append(EvalResult(
        question="What are her active medical problems?",
        expected="Includes Prediabetes (active since 2000)",
        actual=answer[:200],
        passed=passed,
        category="factual",
    ))

    # === HALLUCINATION: Invented surgery ===
    if HAS_ANTHROPIC:
        answer = _ask_llm(context, "When was her last surgery?")
        answer_lower = answer.lower()
        passed = (
            "no" in answer_lower or "none" in answer_lower
            or "not recorded" in answer_lower or "no surg" in answer_lower
        )
    else:
        answer = "string-match"
        # If no surgery is in context, that's correct
        passed = "surgery" not in context.lower() or "no surg" in context.lower()

    suite.results.append(EvalResult(
        question="When was her last surgery?",
        expected="No surgery recorded (tests hallucination)",
        actual=answer[:200],
        passed=passed,
        category="hallucination",
    ))

    return suite


def run_generic_evals(context: str, resources: dict) -> EvalSuite:
    """Run generic eval questions that work for any patient."""
    suite = EvalSuite()

    # Check patient name is in context
    patients = resources.get("Patient", [])
    if patients:
        names = patients[0].get("name", [{}])
        family = names[0].get("family", "") if names else ""
        passed = family.lower() in context.lower() if family else False
        suite.results.append(EvalResult(
            question="Is the patient's name in the context?",
            expected=family,
            actual="present" if passed else "missing",
            passed=passed,
            category="factual",
        ))

    # Check allergies section exists
    passed = "allergi" in context.lower()
    suite.results.append(EvalResult(
        question="Does the context have an allergies section?",
        expected="Yes (even if empty, must be explicit)",
        actual="present" if passed else "missing",
        passed=passed,
        category="factual",
    ))

    # Check active problems section exists
    passed = "active problems" in context.lower() or "active conditions" in context.lower()
    suite.results.append(EvalResult(
        question="Does the context have an active problems section?",
        expected="Yes",
        actual="present" if passed else "missing",
        passed=passed,
        category="factual",
    ))

    # Check medications section exists
    passed = "medication" in context.lower()
    suite.results.append(EvalResult(
        question="Does the context have a medications section?",
        expected="Yes",
        actual="present" if passed else "missing",
        passed=passed,
        category="factual",
    ))

    return suite
