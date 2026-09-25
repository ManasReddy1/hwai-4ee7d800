#!/usr/bin/env python3
"""Streamlit dashboard for visualizing pipeline results.

Run: streamlit run dashboard.py
"""

import json
from pathlib import Path

import streamlit as st
import pandas as pd

OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")


def load_results():
    results_path = OUTPUT_DIR / "pipeline_results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        return json.load(f)


def main():
    st.set_page_config(page_title="Clara Health Context Dashboard", layout="wide")
    st.title("Clara Health Context Dashboard")
    st.caption("FHIR Bundle Compaction Pipeline — Results & Evaluation")

    results = load_results()
    if not results:
        st.error(
            "No pipeline results found. Run `python run_pipeline.py` first."
        )
        return

    # --- Top-level metrics ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Patients Processed", len(results))
    with col2:
        all_pass = all(r["under_900k"] for r in results)
        st.metric("All Under 900K Tokens", "Yes" if all_pass else "No")
    with col3:
        total_evals = sum(r["eval_total"] for r in results)
        total_passed = sum(r["eval_passed"] for r in results)
        st.metric("Eval Pass Rate", f"{total_passed}/{total_evals}")
    with col4:
        st.metric("Token Method", results[0]["token_method"])

    st.divider()

    # --- Compression Overview ---
    st.header("Compression Results")

    df = pd.DataFrame([
        {
            "Patient": r["patient_name"],
            "Raw (bytes)": r["raw_bytes"],
            "Compact (tokens)": r["compact_tokens"],
            "Compression": r["compression_ratio"],
            "Under 900K": r["under_900k"],
            "Eval Pass Rate": r["eval_pass_rate"],
            "Obs Types": r["distinct_obs_types"],
            "Total Obs": r["num_observations"],
        }
        for r in results
    ])

    # Sort by raw size descending
    df = df.sort_values("Raw (bytes)", ascending=False).reset_index(drop=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Token Counts by Patient")
        chart_df = df.head(20).set_index("Patient")[["Compact (tokens)"]]
        st.bar_chart(chart_df)

    with col2:
        st.subheader("Raw Size vs Compact Tokens (Top 20)")
        scatter_df = df.head(20)[["Patient", "Raw (bytes)", "Compact (tokens)"]]
        scatter_df["Raw (KB)"] = scatter_df["Raw (bytes)"] / 1024
        st.scatter_chart(scatter_df, x="Raw (KB)", y="Compact (tokens)")

    st.subheader("All Patients")
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "Raw (bytes)": st.column_config.NumberColumn(format="%d"),
            "Compact (tokens)": st.column_config.NumberColumn(format="%d"),
            "Under 900K": st.column_config.CheckboxColumn(),
        },
    )

    st.divider()

    # --- Eval Details ---
    st.header("Evaluation Results")

    # Merlene detail
    merlene = [r for r in results if "merlene" in r["file"].lower()]
    if merlene:
        m = merlene[0]
        st.subheader(f"Merlene Evals ({m['eval_passed']}/{m['eval_total']} passed)")

        for e in m["eval_details"]:
            icon = "white_check_mark" if e["passed"] else "x"
            with st.expander(f":{icon}: {e['question']}", expanded=not e["passed"]):
                st.write(f"**Category:** {e['category']}")
                st.write(f"**Expected:** {e['expected']}")
                st.write(f"**Actual:** {e['actual']}")

    # Category breakdown
    st.subheader("Results by Category")
    all_evals = []
    for r in results:
        for e in r.get("eval_details", []):
            all_evals.append({
                "Patient": r["patient_name"],
                "Category": e["category"],
                "Question": e["question"],
                "Passed": e["passed"],
            })

    if all_evals:
        eval_df = pd.DataFrame(all_evals)
        cat_summary = eval_df.groupby("Category")["Passed"].agg(["sum", "count"])
        cat_summary.columns = ["Passed", "Total"]
        cat_summary["Rate"] = (cat_summary["Passed"] / cat_summary["Total"] * 100).round(1).astype(str) + "%"
        st.dataframe(cat_summary, use_container_width=True)

    st.divider()

    # --- Patient Deep Dive ---
    st.header("Patient Deep Dive")

    patient_names = [r["patient_name"] for r in results]
    selected = st.selectbox("Select a patient:", patient_names)

    if selected:
        patient = next(r for r in results if r["patient_name"] == selected)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Raw Size", f"{patient['raw_bytes']:,} bytes")
        with col2:
            st.metric("Compact Tokens", f"{patient['compact_tokens']:,}")
        with col3:
            st.metric("Compression", patient["compression_ratio"])

        # Show compact context
        context_file = OUTPUT_DIR / patient["file"].replace(".json", "_compact.md")
        if context_file.exists():
            with st.expander("View Compact Context", expanded=False):
                st.markdown(context_file.read_text()[:5000] + "\n\n*[truncated for display]*")

        # Show RAG index
        rag_file = OUTPUT_DIR / patient["file"].replace(".json", "_rag.json")
        if rag_file.exists():
            with st.expander("Available Lab Tests (RAG Index)", expanded=False):
                labs = json.loads(rag_file.read_text())
                lab_df = pd.DataFrame(labs)
                st.dataframe(lab_df, use_container_width=True)

        # RAG Lookup demo
        st.subheader("RAG Lookup Demo")
        query = st.text_input("Search labs (LOINC code or name):", placeholder="e.g., hba1c, 4548-4, glucose")
        if query:
            from src.fhir_parser import load_bundle, parse_bundle
            from src.rag import PatientIndex

            bundle_path = DATA_DIR / patient["file"]
            if bundle_path.exists():
                bundle = load_bundle(bundle_path)
                parsed = parse_bundle(bundle)
                idx = PatientIndex(parsed["resources"])
                lab_results = idx.lookup_lab(query)
                if lab_results:
                    st.success(f"Found {len(lab_results)} results for '{query}'")
                    res_df = pd.DataFrame(lab_results)
                    st.dataframe(res_df, use_container_width=True)
                else:
                    st.warning(f"No results for '{query}'")

    st.divider()
    st.caption(
        "Built for Clara Health — Founding AI Engineer Interview | "
        "Sai Manas Reddy Kayathi"
    )


if __name__ == "__main__":
    main()
