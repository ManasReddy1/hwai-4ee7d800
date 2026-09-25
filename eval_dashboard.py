"""Interactive Eval Dashboard — Run A/B tests and RAGAS evals dynamically.

Launch: streamlit run eval_dashboard.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.fhir_parser import load_bundle, parse_bundle, extract_patient_demographics
from src.compactor import compact_bundle
from src.llm_evals import (
    run_ab_test,
    run_ragas_eval,
    ab_results_to_dict,
    ragas_results_to_dict,
    generate_questions,
    extract_observations,
)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

st.set_page_config(page_title="Clara Eval Dashboard", page_icon="🔬", layout="wide")


# --- Helpers ---

def get_patient_files():
    """List available patient files."""
    files = []
    for f in sorted(DATA_DIR.glob("*.json")):
        if f.name.startswith("hospital") or f.name.startswith("practitioner"):
            continue
        files.append(f)
    return files


def get_patient_name(path):
    """Extract patient name from a bundle file."""
    bundle = load_bundle(path)
    resources = parse_bundle(bundle)["resources"]
    demo = extract_patient_demographics(resources)
    return demo.get("name", path.stem)


# --- Sidebar ---

st.sidebar.title("Eval Dashboard")
st.sidebar.markdown("**LLM-as-Judge Testing**")

has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
if has_key:
    st.sidebar.success("API Key: Connected")
else:
    st.sidebar.error("API Key: Not set")
    st.sidebar.markdown("Run: `export ANTHROPIC_API_KEY=sk-ant-...`")

mode = st.sidebar.radio("Test Mode", ["Strategy 1: A/B Comparison", "Strategy 2: RAGAS Metrics", "View Saved Results"])

patient_files = get_patient_files()
patient_names = {}
# Build name map from filenames (fast, no parsing)
for f in patient_files:
    name = f.name.split("_")[0]
    patient_names[name] = f

patient_options = sorted(patient_names.keys())

# --- Main Content ---

if mode == "Strategy 1: A/B Comparison":
    st.title("Strategy 1: A/B — Compact Context vs. Raw FHIR")
    st.markdown("""
    **How it works:** The same clinical question is sent to Claude twice — once with our
    compact markdown context, once with the raw FHIR JSON. A separate judge LLM
    scores both answers on 4 metrics.

    **Metrics:** Faithfulness, Answer Relevancy, Completeness, Conciseness
    """)

    selected = st.selectbox("Select patient", patient_options, index=patient_options.index("Merlene950") if "Merlene950" in patient_options else 0)
    patient_path = patient_names[selected]

    # Show what questions will be asked
    with st.expander("Preview auto-generated questions", expanded=False):
        bundle = load_bundle(patient_path)
        resources = parse_bundle(bundle)["resources"]
        demo = extract_patient_demographics(resources)
        questions = generate_questions(resources, demo.get("name", selected))
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1} ({q['category']}):** {q['question']}")
            st.markdown(f"*Ground truth:* {q['ground_truth']}")
            st.markdown("---")

    if not has_key:
        st.warning("Set ANTHROPIC_API_KEY to run tests.")
    elif st.button("Run A/B Test", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Load and prep
        bundle = load_bundle(patient_path)
        resources = parse_bundle(bundle)["resources"]
        demo = extract_patient_demographics(resources)
        questions = generate_questions(resources, demo.get("name", selected))
        total_steps = len(questions) * 10  # rough estimate
        step = [0]

        def ab_progress(msg):
            step[0] += 1
            progress_bar.progress(min(step[0] / max(total_steps, 1), 0.99))
            status_text.text(msg)

        with st.spinner("Running A/B comparison... (this takes 1-2 minutes)"):
            results = run_ab_test(patient_path, progress_callback=ab_progress)

        progress_bar.progress(1.0)
        status_text.text("Done!")

        # Save results
        results_dict = ab_results_to_dict(results)
        save_path = OUTPUT_DIR / f"{selected}_ab_results.json"
        OUTPUT_DIR.mkdir(exist_ok=True)
        save_path.write_text(json.dumps(results_dict, indent=2))

        # Store in session for display
        st.session_state["ab_results"] = results_dict
        st.session_state["ab_patient"] = selected
        st.rerun()

    # Display results if available
    if "ab_results" in st.session_state:
        results = st.session_state["ab_results"]
        patient = st.session_state.get("ab_patient", "")
        st.markdown(f"### Results for {patient}")

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        compact_wins = sum(1 for r in results if r["winner"] == "compact")
        raw_wins = sum(1 for r in results if r["winner"] == "raw")
        ties = sum(1 for r in results if r["winner"] == "tie")

        col1.metric("Compact Wins", compact_wins, delta=f"+{compact_wins - raw_wins}" if compact_wins > raw_wins else None)
        col2.metric("Raw Wins", raw_wins)
        col3.metric("Ties", ties)

        # Average scores comparison
        st.markdown("### Average Scores by Metric")
        metrics = ["faithfulness", "answer_relevancy", "completeness", "conciseness"]
        comparison_data = []
        for m in metrics:
            compact_scores = [r["scores_compact"].get(m, {}).get("score", 0) for r in results]
            raw_scores = [r["scores_raw"].get(m, {}).get("score", 0) for r in results]
            comparison_data.append({
                "Metric": m.replace("_", " ").title(),
                "Compact Context": round(sum(compact_scores) / len(compact_scores), 3) if compact_scores else 0,
                "Raw FHIR": round(sum(raw_scores) / len(raw_scores), 3) if raw_scores else 0,
            })

        df_comp = pd.DataFrame(comparison_data)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        # Bar chart
        chart_data = df_comp.set_index("Metric")
        st.bar_chart(chart_data)

        # Detailed results per question
        st.markdown("### Question-by-Question Breakdown")
        for i, r in enumerate(results):
            winner_emoji = {"compact": "<<", "raw": ">>", "tie": "=="}[r["winner"]]
            winner_label = {"compact": "COMPACT WINS", "raw": "RAW WINS", "tie": "TIE"}[r["winner"]]

            with st.expander(f"Q{i+1}: {r['question'][:80]}... [{winner_label}]"):
                st.markdown(f"**Ground Truth:** {r['ground_truth']}")
                st.markdown("---")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Compact Answer** (avg: {r['compact_avg']:.2f})")
                    st.info(r["answer_compact"])
                    for metric, data in r["scores_compact"].items():
                        st.markdown(f"- **{metric}**: {data['score']:.2f} — {data['reasoning']}")

                with c2:
                    st.markdown(f"**Raw Answer** (avg: {r['raw_avg']:.2f})")
                    st.info(r["answer_raw"])
                    for metric, data in r["scores_raw"].items():
                        st.markdown(f"- **{metric}**: {data['score']:.2f} — {data['reasoning']}")


elif mode == "Strategy 2: RAGAS Metrics":
    st.title("Strategy 2: RAGAS-Inspired Automated Eval")
    st.markdown("""
    **How it works:** Auto-generate questions from raw FHIR data, answer using
    compact context, then score on 4 RAGAS metrics using LLM-as-judge.

    **Metrics:** Faithfulness, Context Recall, Answer Correctness, Hallucination Score
    """)

    # Patient selection — single or multi
    eval_mode = st.radio("Evaluation scope", ["Single Patient", "Multi-Patient (batch)"], horizontal=True)

    if eval_mode == "Single Patient":
        selected = st.selectbox("Select patient", patient_options,
                                index=patient_options.index("Merlene950") if "Merlene950" in patient_options else 0,
                                key="ragas_patient")
        selected_paths = [patient_names[selected]]
    else:
        num = st.slider("Number of patients to evaluate", 2, min(20, len(patient_options)), 5)
        # Pick evenly spaced patients for diversity
        step_size = max(1, len(patient_options) // num)
        selected_list = patient_options[::step_size][:num]
        st.markdown(f"**Selected:** {', '.join(selected_list)}")
        selected_paths = [patient_names[s] for s in selected_list]

    # Preview questions for first patient
    with st.expander("Preview questions for first patient"):
        first_path = selected_paths[0]
        bundle = load_bundle(first_path)
        resources = parse_bundle(bundle)["resources"]
        demo = extract_patient_demographics(resources)
        questions = generate_questions(resources, demo.get("name", "Patient"))
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1} ({q['category']}):** {q['question']}")
            st.markdown(f"*Ground truth:* {q['ground_truth']}")
            st.markdown("---")

    if not has_key:
        st.warning("Set ANTHROPIC_API_KEY to run tests.")
    elif st.button("Run RAGAS Eval", type="primary"):
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for pi, path in enumerate(selected_paths):
            pname = path.name.split("_")[0]
            status_text.text(f"Patient {pi+1}/{len(selected_paths)}: {pname}")
            progress_bar.progress(pi / len(selected_paths))

            def ragas_progress(msg):
                status_text.text(f"[{pname}] {msg}")

            results = run_ragas_eval(path, progress_callback=ragas_progress)
            all_results.extend(results)

        progress_bar.progress(1.0)
        status_text.text("Done!")

        # Save results
        results_dict = ragas_results_to_dict(all_results)
        save_path = OUTPUT_DIR / "ragas_results.json"
        OUTPUT_DIR.mkdir(exist_ok=True)
        save_path.write_text(json.dumps(results_dict, indent=2))

        st.session_state["ragas_results"] = results_dict
        st.rerun()

    # Display RAGAS results
    if "ragas_results" in st.session_state:
        results = st.session_state["ragas_results"]
        st.markdown("### RAGAS Scorecard")

        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        avg_faith = sum(r["faithfulness"] for r in results) / len(results)
        avg_recall = sum(r["context_recall"] for r in results) / len(results)
        avg_correct = sum(r["answer_correctness"] for r in results) / len(results)
        avg_halluc = sum(r["hallucination_score"] for r in results) / len(results)

        col1.metric("Faithfulness", f"{avg_faith:.2f}", help="Are all claims grounded in context?")
        col2.metric("Context Recall", f"{avg_recall:.2f}", help="Does context have the needed info?")
        col3.metric("Answer Correctness", f"{avg_correct:.2f}", help="Does answer match ground truth?")
        col4.metric("Hallucination", f"{avg_halluc:.2f}", delta=f"{-avg_halluc:.2f}", delta_color="inverse", help="Lower is better")

        # Overall score
        avg_overall = sum(r["overall"] for r in results) / len(results)
        st.markdown(f"### Overall RAGAS Score: **{avg_overall:.2f}** / 1.00")

        # By category breakdown
        st.markdown("### Scores by Question Category")
        categories = set(r.get("category", "unknown") for r in results)
        cat_data = []
        for cat in sorted(categories):
            cat_results = [r for r in results if r.get("category") == cat]
            cat_data.append({
                "Category": cat.title(),
                "Count": len(cat_results),
                "Faithfulness": round(sum(r["faithfulness"] for r in cat_results) / len(cat_results), 3),
                "Context Recall": round(sum(r["context_recall"] for r in cat_results) / len(cat_results), 3),
                "Correctness": round(sum(r["answer_correctness"] for r in cat_results) / len(cat_results), 3),
                "Hallucination": round(sum(r["hallucination_score"] for r in cat_results) / len(cat_results), 3),
                "Overall": round(sum(r["overall"] for r in cat_results) / len(cat_results), 3),
            })
        st.dataframe(pd.DataFrame(cat_data), use_container_width=True, hide_index=True)

        # Per-patient breakdown (if multi)
        patients = set(r["patient_name"] for r in results)
        if len(patients) > 1:
            st.markdown("### Scores by Patient")
            patient_data = []
            for pname in sorted(patients):
                p_results = [r for r in results if r["patient_name"] == pname]
                patient_data.append({
                    "Patient": pname,
                    "Questions": len(p_results),
                    "Faithfulness": round(sum(r["faithfulness"] for r in p_results) / len(p_results), 3),
                    "Context Recall": round(sum(r["context_recall"] for r in p_results) / len(p_results), 3),
                    "Correctness": round(sum(r["answer_correctness"] for r in p_results) / len(p_results), 3),
                    "Hallucination": round(sum(r["hallucination_score"] for r in p_results) / len(p_results), 3),
                    "Overall": round(sum(r["overall"] for r in p_results) / len(p_results), 3),
                })
            df_patients = pd.DataFrame(patient_data)
            st.dataframe(df_patients, use_container_width=True, hide_index=True)

            # Chart
            chart_df = df_patients.set_index("Patient")[["Faithfulness", "Context Recall", "Correctness", "Overall"]]
            st.bar_chart(chart_df)

        # Detailed question view
        st.markdown("### Question Details")
        for i, r in enumerate(results):
            score_color = "green" if r["overall"] >= 0.8 else "orange" if r["overall"] >= 0.5 else "red"
            with st.expander(f"Q{i+1} [{r.get('category', '?')}]: {r['question'][:80]}... (overall: {r['overall']:.2f})"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Patient:** {r['patient_name']}")
                    st.markdown(f"**Ground Truth:** {r['ground_truth']}")
                    st.markdown(f"**Answer:** {r['answer']}")
                with c2:
                    st.markdown("**Scores:**")
                    st.markdown(f"- Faithfulness: **{r['faithfulness']:.2f}** — {r['details'].get('faithfulness_reasoning', '')}")
                    st.markdown(f"- Context Recall: **{r['context_recall']:.2f}** — {r['details'].get('context_recall_reasoning', '')}")
                    st.markdown(f"- Correctness: **{r['answer_correctness']:.2f}** — {r['details'].get('correctness_reasoning', '')}")
                    st.markdown(f"- Hallucination: **{r['hallucination_score']:.2f}** — {r['details'].get('hallucination_reasoning', '')}")


elif mode == "View Saved Results":
    st.title("Saved Eval Results")

    # Check for saved files
    ab_files = list(OUTPUT_DIR.glob("*_ab_results.json"))
    ragas_file = OUTPUT_DIR / "ragas_results.json"

    if ab_files:
        st.markdown("### A/B Test Results")
        for f in ab_files:
            patient = f.stem.replace("_ab_results", "")
            with st.expander(f"A/B Results: {patient}"):
                data = json.loads(f.read_text())
                compact_wins = sum(1 for r in data if r["winner"] == "compact")
                raw_wins = sum(1 for r in data if r["winner"] == "raw")
                ties = sum(1 for r in data if r["winner"] == "tie")
                st.markdown(f"**Compact wins: {compact_wins} | Raw wins: {raw_wins} | Ties: {ties}**")

                for i, r in enumerate(data):
                    st.markdown(f"**Q{i+1}:** {r['question']}")
                    st.markdown(f"Winner: **{r['winner'].upper()}** | Compact: {r['compact_avg']:.2f} | Raw: {r['raw_avg']:.2f}")
                    st.markdown("---")

    if ragas_file.exists():
        st.markdown("### RAGAS Results")
        data = json.loads(ragas_file.read_text())
        avg_overall = sum(r["overall"] for r in data) / len(data) if data else 0
        st.markdown(f"**Overall RAGAS Score: {avg_overall:.2f}** across {len(data)} questions")

        df = pd.DataFrame(data)[["patient_name", "category", "question", "faithfulness",
                                   "context_recall", "answer_correctness", "hallucination_score", "overall"]]
        st.dataframe(df, use_container_width=True, hide_index=True)

    if not ab_files and not ragas_file.exists():
        st.info("No saved results yet. Run a test from Strategy 1 or Strategy 2 first.")
