"""LLM-as-Judge Eval System with RAGAS-inspired metrics.

Two strategies:
1. Strategy A — Pairwise A/B: Same question, compact vs raw context, judge scores both
2. Strategy B — RAGAS Automated: Auto-generate questions from FHIR data, score on
   Faithfulness, Context Recall, Answer Correctness, Hallucination

All scoring uses Claude as the judge via the Anthropic API.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from anthropic import Anthropic

from src.fhir_parser import (
    extract_allergies,
    extract_conditions,
    extract_medications,
    extract_observations,
    extract_patient_demographics,
    load_bundle,
    parse_bundle,
)
from src.compactor import compact_bundle
from src.rag import PatientIndex


# --- Data Classes ---

@dataclass
class JudgeScore:
    """Score from a single LLM judge evaluation."""
    metric: str          # e.g., "faithfulness", "answer_relevancy"
    score: float         # 0.0 to 1.0
    reasoning: str       # Judge's explanation
    raw_response: str = ""


@dataclass
class ABResult:
    """Result of a single A/B comparison."""
    question: str
    ground_truth: str
    answer_compact: str
    answer_raw: str
    scores_compact: list[JudgeScore] = field(default_factory=list)
    scores_raw: list[JudgeScore] = field(default_factory=list)

    def compact_avg(self) -> float:
        return sum(s.score for s in self.scores_compact) / len(self.scores_compact) if self.scores_compact else 0

    def raw_avg(self) -> float:
        return sum(s.score for s in self.scores_raw) / len(self.scores_raw) if self.scores_raw else 0

    def winner(self) -> str:
        c, r = self.compact_avg(), self.raw_avg()
        if c > r + 0.05:
            return "compact"
        elif r > c + 0.05:
            return "raw"
        return "tie"


@dataclass
class RAGASResult:
    """Result of a single RAGAS-style evaluation."""
    question: str
    ground_truth: str
    answer: str
    patient_name: str
    faithfulness: float = 0.0
    context_recall: float = 0.0
    answer_correctness: float = 0.0
    hallucination_score: float = 0.0
    details: dict = field(default_factory=dict)

    def overall(self) -> float:
        return (self.faithfulness + self.context_recall +
                self.answer_correctness + (1 - self.hallucination_score)) / 4


# --- LLM Helpers ---

def _get_client() -> Anthropic:
    return Anthropic()


def _ask_claude(context: str, question: str, client: Optional[Anthropic] = None) -> str:
    """Ask Claude a clinical question with the given context."""
    if client is None:
        client = _get_client()

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                "You are a clinical assistant. Answer the question based ONLY on the "
                "patient record below. If the information is not in the record, say "
                '"not recorded" or "none recorded". Be concise and specific.\n\n'
                f"<patient_record>\n{context[:80000]}\n</patient_record>\n\n"
                f"Question: {question}\n\nAnswer:"
            ),
        }],
    )
    return response.content[0].text


def _judge_score(answer: str, context: str, question: str, ground_truth: str,
                  metric: str, client: Optional[Anthropic] = None) -> JudgeScore:
    """Have Claude judge an answer on a specific RAGAS metric."""
    if client is None:
        client = _get_client()

    metric_prompts = {
        "faithfulness": (
            "Rate the FAITHFULNESS of the answer on a scale of 0.0 to 1.0.\n"
            "Faithfulness = every claim in the answer can be directly traced to the context.\n"
            "1.0 = every single claim is supported by the context, nothing is invented.\n"
            "0.0 = the answer contains information not found in the context.\n"
            "Check each claim in the answer against the context. If ANY claim is not supported, lower the score."
        ),
        "answer_relevancy": (
            "Rate the ANSWER RELEVANCY on a scale of 0.0 to 1.0.\n"
            "Answer relevancy = how well does the answer actually address the question?\n"
            "1.0 = perfectly answers what was asked, concise, no tangents.\n"
            "0.0 = completely off-topic or refuses to answer when it could."
        ),
        "completeness": (
            "Rate the COMPLETENESS of the answer on a scale of 0.0 to 1.0.\n"
            "Compare the answer to the ground truth. Does the answer capture all key facts?\n"
            "1.0 = answer contains all important facts from the ground truth.\n"
            "0.0 = answer misses all the important facts."
        ),
        "conciseness": (
            "Rate the CONCISENESS of the answer on a scale of 0.0 to 1.0.\n"
            "1.0 = answer is to-the-point with no unnecessary information.\n"
            "0.0 = answer is extremely verbose with lots of irrelevant filler."
        ),
        "context_recall": (
            "Rate the CONTEXT RECALL on a scale of 0.0 to 1.0.\n"
            "Context recall = does the context contain enough information to answer correctly?\n"
            "Compare ground truth facts to what's available in the context.\n"
            "1.0 = all facts needed to answer are present in the context.\n"
            "0.0 = none of the needed facts are in the context."
        ),
        "answer_correctness": (
            "Rate the ANSWER CORRECTNESS on a scale of 0.0 to 1.0.\n"
            "Compare the answer to the ground truth. Are the facts correct?\n"
            "1.0 = answer matches the ground truth exactly.\n"
            "0.0 = answer contradicts the ground truth completely."
        ),
        "hallucination": (
            "Rate the HALLUCINATION level on a scale of 0.0 to 1.0.\n"
            "Hallucination = the answer contains facts NOT present in the context.\n"
            "1.0 = the answer is entirely hallucinated / fabricated.\n"
            "0.0 = every fact in the answer is grounded in the context.\n"
            "Even one invented detail should raise this score above 0."
        ),
    }

    prompt = (
        f"You are an expert evaluator judging the quality of a clinical AI answer.\n\n"
        f"{metric_prompts[metric]}\n\n"
        f"<context>\n{context[:40000]}\n</context>\n\n"
        f"<question>{question}</question>\n\n"
        f"<ground_truth>{ground_truth}</ground_truth>\n\n"
        f"<answer>{answer}</answer>\n\n"
        f"Respond in this exact JSON format:\n"
        f'{{"score": 0.X, "reasoning": "one sentence explanation"}}'
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    try:
        # Extract JSON from response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        return JudgeScore(
            metric=metric,
            score=max(0.0, min(1.0, float(parsed["score"]))),
            reasoning=parsed.get("reasoning", ""),
            raw_response=raw,
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return JudgeScore(metric=metric, score=0.5, reasoning=f"Parse error: {raw[:100]}", raw_response=raw)


# --- Question Generation ---

def generate_questions(resources: dict, patient_name: str) -> list[dict]:
    """Auto-generate clinical questions + ground truth from raw FHIR data."""
    questions = []
    observations = extract_observations(resources)
    demographics = extract_patient_demographics(resources)
    active_conditions, resolved_conditions = extract_conditions(resources)
    medications = extract_medications(resources)
    allergies = extract_allergies(resources)

    # 1. Factual: Latest value of most-recorded lab
    if observations:
        top_labs = sorted(observations.items(), key=lambda x: len(x[1]), reverse=True)[:3]
        for lab_name, readings in top_labs:
            if readings and readings[-1].get("value") is not None:
                latest = readings[-1]
                questions.append({
                    "question": f"What is {patient_name}'s latest {lab_name.split('[')[0].strip()} result?",
                    "ground_truth": f"{latest['value']} {latest.get('unit', '')} on {latest['date'][:10]}",
                    "category": "factual",
                })
                break  # just the top one

    # 2. Factual: Active conditions
    if active_conditions:
        condition_names = [c["display"] for c in active_conditions[:5]]
        questions.append({
            "question": f"What are {patient_name}'s active medical problems?",
            "ground_truth": ", ".join(condition_names),
            "category": "factual",
        })

    # 3. Factual: Current medications
    current_meds = [m for m in medications if m.get("status") == "active"]
    if current_meds:
        med_names = [m["display"][:60] for m in current_meds[:5]]
        questions.append({
            "question": f"What medications is {patient_name} currently taking?",
            "ground_truth": ", ".join(med_names),
            "category": "factual",
        })

    # 4. Trend: Lab with multiple readings
    if observations:
        for lab_name, readings in observations.items():
            if len(readings) >= 5 and readings[-1].get("value") is not None:
                try:
                    vals = [float(r["value"]) for r in readings[-5:] if r.get("value") is not None]
                    if len(vals) >= 3:
                        first, last = vals[0], vals[-1]
                        if last > first * 1.05:
                            trend = "rising"
                        elif last < first * 0.95:
                            trend = "falling"
                        else:
                            trend = "stable"
                        questions.append({
                            "question": f"How has {patient_name}'s {lab_name.split('[')[0].strip()} been trending over time?",
                            "ground_truth": f"Trend is {trend}. Latest: {readings[-1]['value']} {readings[-1].get('unit', '')}. "
                                          f"Total readings: {len(readings)}.",
                            "category": "trend",
                        })
                        break
                except (ValueError, TypeError):
                    continue

    # 5. Hallucination: Allergies (adversarial)
    if len(allergies) == 0:
        questions.append({
            "question": f"What is {patient_name} allergic to?",
            "ground_truth": "No allergies recorded / none known",
            "category": "hallucination",
        })
    else:
        allergy_names = [a["display"] for a in allergies]
        questions.append({
            "question": f"What is {patient_name} allergic to?",
            "ground_truth": ", ".join(allergy_names),
            "category": "factual",
        })

    # 6. Hallucination: Invented medication
    questions.append({
        "question": f"Is {patient_name} currently taking amoxicillin?",
        "ground_truth": "Yes" if any("amoxicillin" in m.get("display", "").lower() and m.get("status") == "active" for m in medications) else "No, not currently taking amoxicillin",
        "category": "hallucination",
    })

    return questions


# --- Strategy 1: Pairwise A/B ---

def run_ab_test(patient_path, progress_callback=None) -> list[ABResult]:
    """Run pairwise A/B comparison: compact context vs raw FHIR context."""
    client = _get_client()
    bundle = load_bundle(patient_path)
    parsed = parse_bundle(bundle)
    resources = parsed["resources"]
    demographics = extract_patient_demographics(resources)
    patient_name = demographics.get("name", "Unknown")

    # Build both contexts
    compact_context = compact_bundle(resources)
    raw_context = json.dumps(bundle, indent=1)[:80000]  # truncate raw to fit in context

    # Generate questions
    questions = generate_questions(resources, patient_name)

    results = []
    ab_metrics = ["faithfulness", "answer_relevancy", "completeness", "conciseness"]

    for i, q in enumerate(questions):
        if progress_callback:
            progress_callback(f"Q{i+1}/{len(questions)}: {q['question'][:60]}...")

        # Get answers from both contexts
        answer_compact = _ask_claude(compact_context, q["question"], client)
        time.sleep(0.5)  # rate limit
        answer_raw = _ask_claude(raw_context, q["question"], client)
        time.sleep(0.5)

        result = ABResult(
            question=q["question"],
            ground_truth=q["ground_truth"],
            answer_compact=answer_compact,
            answer_raw=answer_raw,
        )

        # Judge both answers on all metrics
        for metric in ab_metrics:
            score_compact = _judge_score(answer_compact, compact_context, q["question"], q["ground_truth"], metric, client)
            time.sleep(0.3)
            score_raw = _judge_score(answer_raw, raw_context, q["question"], q["ground_truth"], metric, client)
            time.sleep(0.3)
            result.scores_compact.append(score_compact)
            result.scores_raw.append(score_raw)

        results.append(result)

    return results


# --- Strategy 2: RAGAS Automated Pipeline ---

def run_ragas_eval(patient_path, progress_callback=None) -> list[RAGASResult]:
    """Run RAGAS-inspired evaluation on compact context."""
    client = _get_client()
    bundle = load_bundle(patient_path)
    parsed = parse_bundle(bundle)
    resources = parsed["resources"]
    demographics = extract_patient_demographics(resources)
    patient_name = demographics.get("name", "Unknown")

    compact_context = compact_bundle(resources)
    questions = generate_questions(resources, patient_name)

    results = []

    for i, q in enumerate(questions):
        if progress_callback:
            progress_callback(f"Q{i+1}/{len(questions)}: {q['question'][:60]}...")

        # Get answer from compact context
        answer = _ask_claude(compact_context, q["question"], client)
        time.sleep(0.5)

        # Score on 4 RAGAS metrics
        faithfulness = _judge_score(answer, compact_context, q["question"], q["ground_truth"], "faithfulness", client)
        time.sleep(0.3)
        context_recall = _judge_score(answer, compact_context, q["question"], q["ground_truth"], "context_recall", client)
        time.sleep(0.3)
        correctness = _judge_score(answer, compact_context, q["question"], q["ground_truth"], "answer_correctness", client)
        time.sleep(0.3)
        hallucination = _judge_score(answer, compact_context, q["question"], q["ground_truth"], "hallucination", client)
        time.sleep(0.3)

        result = RAGASResult(
            question=q["question"],
            ground_truth=q["ground_truth"],
            answer=answer,
            patient_name=patient_name,
            faithfulness=faithfulness.score,
            context_recall=context_recall.score,
            answer_correctness=correctness.score,
            hallucination_score=hallucination.score,
            details={
                "category": q["category"],
                "faithfulness_reasoning": faithfulness.reasoning,
                "context_recall_reasoning": context_recall.reasoning,
                "correctness_reasoning": correctness.reasoning,
                "hallucination_reasoning": hallucination.reasoning,
            },
        )
        results.append(result)

    return results


# --- Serialization ---

def ab_results_to_dict(results: list[ABResult]) -> list[dict]:
    """Convert A/B results to serializable dicts."""
    return [
        {
            "question": r.question,
            "ground_truth": r.ground_truth,
            "answer_compact": r.answer_compact,
            "answer_raw": r.answer_raw,
            "winner": r.winner(),
            "compact_avg": round(r.compact_avg(), 3),
            "raw_avg": round(r.raw_avg(), 3),
            "scores_compact": {s.metric: {"score": round(s.score, 3), "reasoning": s.reasoning} for s in r.scores_compact},
            "scores_raw": {s.metric: {"score": round(s.score, 3), "reasoning": s.reasoning} for s in r.scores_raw},
        }
        for r in results
    ]


def ragas_results_to_dict(results: list[RAGASResult]) -> list[dict]:
    """Convert RAGAS results to serializable dicts."""
    return [
        {
            "question": r.question,
            "ground_truth": r.ground_truth,
            "answer": r.answer,
            "patient_name": r.patient_name,
            "faithfulness": round(r.faithfulness, 3),
            "context_recall": round(r.context_recall, 3),
            "answer_correctness": round(r.answer_correctness, 3),
            "hallucination_score": round(r.hallucination_score, 3),
            "overall": round(r.overall(), 3),
            "category": r.details.get("category", ""),
            "details": r.details,
        }
        for r in results
    ]
