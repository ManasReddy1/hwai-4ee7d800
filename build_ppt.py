#!/usr/bin/env python3
"""Build the presentation PPT for Clara onsite interview."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import json
from pathlib import Path


def add_title_slide(prs, title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
    # Title
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(2), Inches(8.4), Inches(1.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x2C, 0x5F, 0x8A)
    p.alignment = PP_ALIGN.CENTER

    p2 = tf.add_paragraph()
    p2.text = subtitle
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    p2.alignment = PP_ALIGN.CENTER
    return slide


def add_content_slide(prs, title, bullets, subtitle=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank

    # Title bar
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x2C, 0x5F, 0x8A)

    if subtitle:
        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.font.size = Pt(14)
        p2.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    # Content
    txBox2 = slide.shapes.add_textbox(Inches(0.7), Inches(1.3), Inches(8.6), Inches(5.5))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True

    for i, bullet in enumerate(bullets):
        if i == 0:
            p = tf2.paragraphs[0]
        else:
            p = tf2.add_paragraph()

        if bullet.startswith("##"):
            p.text = bullet.replace("## ", "")
            p.font.size = Pt(16)
            p.font.bold = True
            p.font.color.rgb = RGBColor(0x3A, 0x7C, 0xB8)
            p.space_before = Pt(12)
        elif bullet.startswith("  -"):
            p.text = bullet.strip("  -").strip()
            p.font.size = Pt(15)
            p.level = 1
            p.space_before = Pt(4)
        elif bullet.startswith("**"):
            text = bullet.replace("**", "")
            p.text = text
            p.font.size = Pt(16)
            p.font.bold = True
            p.space_before = Pt(8)
        else:
            p.text = bullet
            p.font.size = Pt(16)
            p.space_before = Pt(6)

    return slide


def add_two_column_slide(prs, title, left_bullets, right_bullets, left_title="", right_title=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Title
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.7))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x2C, 0x5F, 0x8A)

    # Left column
    txL = slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(4.3), Inches(5.5))
    tfL = txL.text_frame
    tfL.word_wrap = True
    if left_title:
        p = tfL.paragraphs[0]
        p.text = left_title
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0x3A, 0x7C, 0xB8)
        for b in left_bullets:
            p2 = tfL.add_paragraph()
            p2.text = b
            p2.font.size = Pt(14)
            p2.space_before = Pt(4)
    else:
        for i, b in enumerate(left_bullets):
            p = tfL.paragraphs[0] if i == 0 else tfL.add_paragraph()
            p.text = b
            p.font.size = Pt(14)
            p.space_before = Pt(4)

    # Right column
    txR = slide.shapes.add_textbox(Inches(5.2), Inches(1.2), Inches(4.3), Inches(5.5))
    tfR = txR.text_frame
    tfR.word_wrap = True
    if right_title:
        p = tfR.paragraphs[0]
        p.text = right_title
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0x3A, 0x7C, 0xB8)
        for b in right_bullets:
            p2 = tfR.add_paragraph()
            p2.text = b
            p2.font.size = Pt(14)
            p2.space_before = Pt(4)
    else:
        for i, b in enumerate(right_bullets):
            p = tfR.paragraphs[0] if i == 0 else tfR.add_paragraph()
            p.text = b
            p.font.size = Pt(14)
            p.space_before = Pt(4)

    return slide


def build():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    # ---- SLIDE 1: Title ----
    add_title_slide(
        prs,
        "Health Context Challenge",
        "Turning 52 MB of FHIR Records into Tight, Eval-Proven AI Context\n\n"
        "Sai Manas Reddy Kayathi\nClara Health — Founding AI Engineer"
    )

    # ---- SLIDE 2: The Problem ----
    add_content_slide(prs, "The Problem", [
        "Patient medical records (FHIR bundles) are massive — up to 52 MB of JSON",
        "That's ~10 million tokens. Claude's window is 1 million.",
        "But even if it fit, more context = worse performance (proven).",
        "",
        "**The Goal:**",
        "Compress intelligently to <900K tokens without losing clinical answers.",
        "Build evals that prove it works.",
        "Build retrieval for anything that didn't fit.",
    ])

    # ---- SLIDE 3: Why Less Is More ----
    add_content_slide(prs, "Why Smaller Context = Better Answers", [
        '**"Lost in the Middle" (Liu et al., TACL 2024)**',
        "LLMs attend best to the beginning and end of context.",
        "30%+ accuracy drop for information placed in the middle.",
        "",
        '**"Context Rot" (Chroma, 2025)**',
        "Tested 18 frontier models (Claude, GPT-4.1, Gemini 2.5).",
        "Every model degrades at every context length increment.",
        "Even a 1M-token window rots at 50K tokens.",
        "",
        "Compression is not a compromise — it improves quality.",
    ])

    # ---- SLIDE 4: The Data ----
    add_two_column_slide(
        prs, "The Data: 109 Synthea FHIR Bundles",
        [
            "109 patient bundles (106 KB to 49.9 MB)",
            "2 shared files (hospitals + practitioners)",
            "9 patients exceed 900K tokens raw",
            "21 patients have recorded allergies",
            "9 patients are deceased",
            "Latest encounter: 2026-08-16",
        ],
        [
            "42% of bytes = billing (zero clinical value)",
            "98% of note lines are duplicates",
            "Notes stored TWICE (DocRef + DiagReport)",
            "Same urinalysis panel repeated 211x",
            "Only 84 distinct observation types out of 9,610",
            "Provenance: 1.2 MB of metadata for one patient",
        ],
        left_title="Key Numbers",
        right_title="Key Waste Found",
    )

    # ---- SLIDE 5: What I Drop ----
    add_two_column_slide(
        prs, "Compaction Strategy: What Gets Dropped vs Kept",
        [
            "Claim (billing) — 9% of bytes, 0% clinical",
            "ExplanationOfBenefit — 33% of bytes, 0% clinical",
            "Provenance — metadata, 3% of bytes",
            "Duplicate notes in DiagnosticReport",
            "MedicationAdministration (redundant)",
            "SupplyDelivery (low value)",
            "",
            "Total dropped: ~47% of raw bytes",
        ],
        [
            "Patient demographics — always",
            "Allergies — always (explicit absence!)",
            "Active conditions — always",
            "Medications — deduplicated",
            "Observations — collapsed to latest + trend",
            "Clinical notes — line-level deduplication",
            "Procedures, immunizations — keep unique",
        ],
        left_title="Dropped Entirely",
        right_title="Kept (Smart Compression)",
    )

    # ---- SLIDE 6: Output Structure ----
    add_content_slide(prs, "Output: IPS-Structured Markdown", [
        "Structure follows the International Patient Summary (IPS) standard:",
        "",
        "1. Patient demographics (name, DOB, gender)",
        "2. Allergies — explicit 'No allergies recorded' if none",
        "3. Active problems (conditions with onset dates)",
        "4. Medications (current + historical, deduplicated)",
        "5. Lab results table (latest value, trend, reading count)",
        "6. Immunizations, procedures",
        "7. Resolved problems",
        "8. Clinical notes (deduplicated unique lines)",
        "9. Encounters summary",
        "",
        "**Why Markdown?** 3-5x more token-efficient than JSON.",
        "**Why this order?** Matches clinician chart review workflow + places critical info at start.",
    ])

    # ---- SLIDE 7: Compression Results ----
    add_content_slide(prs, "Compression Results", [
        "**Largest patient (Cole117):**",
        "  - Raw: 49.9 MB (~9.7M tokens) → Compact: ~16K tokens",
        "  - Compression: 99.8%",
        "",
        "**Median patient (Susan422):**",
        "  - Raw: 2.3 MB (~420K tokens) → Compact: ~5K tokens",
        "  - Compression: 98.8%",
        "",
        "**Eval patient (Merlene):**",
        "  - Raw: 1.6 MB (~403K tokens) → Compact: ~5.5K tokens",
        "  - Compression: 98.6%",
        "",
        "**All 109 patients fit under 900K.** Max compact = ~16K tokens.",
        "The 900K cap is easy. The real question: how small before evals fail?",
    ])

    # ---- SLIDE 8: Merlene Deep Dive ----
    add_content_slide(prs, "Merlene: The Eval Patient", [
        "8 HbA1c readings, 2016-2025:",
        "  - 2016: 5.99 → 2017: 5.88 → 2018: 6.31 → 2020: 6.29",
        "  - 2022: 6.33 → 2023: 6.32 → 2024: 6.15 → 2025: 6.31",
        "",
        "Latest HbA1c: 6.31 on 2025-09-29",
        "Trend: Rose ~5.9→6.3 by 2018, then plateaued. Never crossed 6.5.",
        "Active diagnosis: Prediabetes (since 2000-11-20)",
        "Allergies: NONE (0 AllergyIntolerance resources)",
        "",
        "**Why this patient is a great eval case:**",
        "  - Tests exact retrieval (latest HbA1c)",
        "  - Tests trend reasoning (is it getting worse?)",
        "  - Tests hallucination resistance (allergies = nothing)",
    ])

    # ---- SLIDE 9: Eval Framework ----
    add_content_slide(prs, "Eval Framework: Three Dimensions", [
        "## 1. Factual Accuracy (Exact Match)",
        "  - Q: Latest HbA1c? A: 6.31 on 2025-09-29",
        "  - Q: How many readings? A: 8",
        "  - Q: Active diagnoses? A: Includes Prediabetes",
        "",
        "## 2. Trend & Reasoning (LLM-as-Judge)",
        "  - Q: Is blood sugar getting worse?",
        "  - Must mention prediabetes + stable/plateaued trend",
        "",
        "## 3. Hallucination Resistance (Adversarial)",
        "  - Q: What is she allergic to? A: Nothing recorded",
        "  - Q: When was her last surgery? A: No surgery recorded",
        "  - Tests that model does NOT invent information",
    ], subtitle="Why not ROUGE/BERTScore? They correlate poorly with clinical faithfulness (Van Veen, Nature Med 2024)")

    # ---- SLIDE 10: Eval Results ----
    add_content_slide(prs, "Eval Results", [
        "**Merlene (6 evals):**",
        "  - Factual: Latest HbA1c, reading count, active problems",
        "  - Trend: Blood sugar trajectory + diagnosis link",
        "  - Hallucination: Allergy check, surgery check",
        "",
        "**All 109 patients (generic evals):**",
        "  - Patient name present in context",
        "  - Allergies section exists (explicit absence)",
        "  - Active problems section exists",
        "  - Medications section exists",
        "",
        "Token counting: chars/4 estimate (no API key set).",
        "With ANTHROPIC_API_KEY, evals call Claude for real LLM grading.",
        "",
        "Run: python run_pipeline.py",
        "Dashboard: streamlit run dashboard.py",
    ])

    # ---- SLIDE 11: RAG System ----
    add_content_slide(prs, "RAG / Lookup System", [
        "**Design choice: Code-based lookup, not vector database.**",
        "",
        "FHIR data is already coded with standard terminologies:",
        "  - Labs → LOINC codes (e.g., 4548-4 = HbA1c)",
        "  - Conditions → SNOMED codes",
        "  - Medications → RxNorm codes",
        "",
        "Lookup by LOINC code = O(1), deterministic, zero infrastructure.",
        "",
        "**Generalizes to arbitrary labs automatically:**",
        '  - index.lookup_lab("4548-4")  → all HbA1c readings',
        '  - index.lookup_lab("glucose") → all glucose results',
        '  - index.lookup_all_lab_codes() → list all available tests',
        "",
        "A vector DB adds complexity without benefit for structured data.",
        "If needed: add vector search as supplementary path for fuzzy queries.",
    ])

    # ---- SLIDE 12: Architecture ----
    add_content_slide(prs, "Architecture Overview", [
        "Raw FHIR Bundle (.json)",
        "     │",
        "     ▼",
        "  fhir_parser.py — Parse, categorize by resource type",
        "     │",
        "     ├──▶ compactor.py — Drop billing, dedupe, collapse → Markdown",
        "     │        └──▶ *_compact.md (compact context, <900K tokens)",
        "     │",
        "     ├──▶ rag.py — Build LOINC/SNOMED lookup index",
        "     │        └──▶ *_rag.json (searchable lab index)",
        "     │",
        "     └──▶ evals.py — Run factual + trend + hallucination tests",
        "              └──▶ pipeline_results.json",
        "",
        "  dashboard.py — Streamlit UI showing all results",
        "  run_pipeline.py — Orchestrates everything",
    ])

    # ---- SLIDE 13: Why Deterministic ----
    add_content_slide(prs, "Key Design Decision: Fully Deterministic", [
        "The entire compaction pipeline uses ZERO LLM calls.",
        "",
        "**Why?**",
        "  - Reproducible: same input → same output, every time",
        "  - Free: no API costs during compaction",
        "  - Fast: processes 52 MB patient in seconds",
        "  - No hallucination risk in the compaction step itself",
        "  - Testable: evals are meaningful because output is stable",
        "",
        "**When would I add LLM?**",
        "  - Summarizing very long unique clinical note text",
        "  - Generating problem-specific summaries for targeted queries",
        "  - Only when deterministic compression hits diminishing returns",
    ])

    # ---- SLIDE 14: Trade-offs ----
    add_two_column_slide(
        prs, "Trade-offs Considered",
        [
            "Markdown vs JSON output",
            "  → Markdown: 3-5x fewer tokens",
            "",
            "IPS order vs chronological",
            "  → IPS: matches clinician workflow",
            "",
            "Latest value + trend vs all values",
            "  → Collapsed: 9,610 obs → 84 rows",
            "",
            "Code lookup vs vector DB",
            "  → Code: exact, deterministic, zero infra",
        ],
        [
            "Plain Python vs dagster",
            "  → Plain: 109 files is trivially small",
            "  → dagster for production scale",
            "",
            "Deterministic vs LLM compaction",
            "  → Deterministic: reproducible, free, safe",
            "",
            "Line dedup vs latest-note-only",
            "  → Line dedup: keeps all unique content",
            "",
            "chars/4 vs Anthropic token count",
            "  → chars/4 for now; API when key is set",
        ],
        left_title="What I Chose (and Why)",
        right_title="Other Decisions",
    )

    # ---- SLIDE 15: Production Evolution ----
    add_content_slide(prs, "What Changes at Production Scale", [
        "**dagster pipeline** with partitioned assets per patient.",
        "  - Sensor watches for new FHIR data, triggers reprocessing",
        "  - Each asset: raw → parsed → compacted → indexed → evaled",
        "",
        "**Incremental updates** — don't reprocess entire bundle for new records.",
        "",
        "**LLM note summarization** — for patients with extensive unique notes.",
        "",
        "**Vector search fallback** — for fuzzy natural-language queries.",
        "",
        "**Caching** — compact contexts and RAG indices, invalidate on new data.",
        "",
        "**Cost monitoring** — track token usage per patient, per query.",
        "",
        "**Continuous evals** — run on every compaction change, track quality over time.",
    ])

    # ---- SLIDE 16: Safety ----
    add_content_slide(prs, "Clinical Safety", [
        '**"Absent and Unknown" (IPS Convention):**',
        '  - Empty allergy section ≠ "no allergies"',
        '  - We explicitly write "No allergies recorded"',
        "  - Prevents the #1 hallucination failure mode",
        "",
        "**Never-drop resources:** Allergies, active conditions, current medications.",
        "",
        "**Hallucination eval** tests that the model does NOT invent:",
        "  - Allergies when there are none",
        "  - Surgeries when there are none",
        "  - Values not present in the record",
        "",
        "**Grounding prompt:** Model must quote source data before answering.",
        "",
        "**Human-in-the-loop:** Clara's providers review every AI decision.",
    ])

    # ---- SLIDE 17: Research Foundation ----
    add_content_slide(prs, "Research Foundation", [
        "**Context Optimization:**",
        "  - Lost in the Middle (Liu et al., TACL 2024)",
        "  - Context Rot (Chroma, 2025) — 18 models tested",
        "",
        "**Clinical Summarization:**",
        "  - Van Veen et al. (Nature Medicine, 2024)",
        "  - LLMs outperform experts but hallucinate at 1.47% rate",
        "  - ROUGE/BERTScore do NOT reliably detect clinical errors",
        "",
        "**Standards:**",
        "  - HL7 FHIR IPS v2.0 — section structure template",
        "  - USCDI v4 — minimum data classes",
        "",
        "**Benchmarks:**",
        "  - FHIR-AgentBench (2025) — temporal reasoning is hardest",
        "  - MedAgentBench (2025) — multi-step workflows fail most",
    ])

    # ---- SLIDE 18: Live Demo ----
    add_content_slide(prs, "Live Demo", [
        "**1. Run the pipeline:**",
        "   python run_pipeline.py Merlene",
        "",
        "**2. See the compact context:**",
        "   output/Merlene*_compact.md",
        "",
        "**3. See the eval results:**",
        "   output/pipeline_results.json",
        "",
        "**4. Launch the dashboard:**",
        "   streamlit run dashboard.py",
        "",
        "**5. Try RAG lookup:**",
        '   Search for "hba1c" or "glucose" in the dashboard',
    ])

    # ---- SLIDE 19: Summary ----
    add_content_slide(prs, "Summary", [
        "**Problem:** 52 MB FHIR bundles → need <900K token context for Claude.",
        "",
        "**Insight:** 47% of data is billing waste, 98% of notes are duplicates,",
        "and smaller context actually improves LLM accuracy (context rot).",
        "",
        "**Solution:** Deterministic compaction → IPS-structured markdown.",
        "99.8% compression for the largest patient. Zero LLM calls.",
        "",
        "**Evals:** Factual accuracy + trend reasoning + hallucination resistance.",
        "Ground-truth extracted from raw FHIR data.",
        "",
        "**RAG:** LOINC-code lookup for arbitrary lab retrieval. Zero infrastructure.",
        "",
        "**Key trade-off:** Simplicity over sophistication. Every line is explainable.",
    ])

    # ---- SLIDE 20: Clara App Feedback ----
    add_content_slide(prs, "Clara App (app.askclara.com) — Feedback", [
        "**What I liked:**",
        "  - Clean ChatGPT-style conversational UI — familiar, low friction",
        "  - Smart quick-action chips (Prescriptions, Analyze Labs, Symptoms, etc.)",
        "  - 'Connect my records' CTA is clear — bridges the cold-start problem",
        "  - Left sidebar mirrors clinical workflow: Chat, Prescriptions, Labs, Appointments",
        "  - Safety disclaimer at the bottom: 'Clara is AI, not a human clinician'",
        "",
        "**Ideas for improvement:**",
        "  - Show patient context status — 'Records connected: 3 sources' builds trust",
        "  - Add a 'Recent labs summary' widget above the chat for at-a-glance info",
        "  - Chip suggestions could be contextual (e.g., after connecting records,",
        "    show 'Review my latest bloodwork' instead of generic 'Analyze labs')",
        "  - Conversation history in sidebar would help continuity across sessions",
        "  - Consider a confidence indicator on AI responses — high/medium/low",
        "    with 'Ask your provider' nudge for low-confidence answers",
    ])

    # ---- SLIDE 21: Thank You ----
    add_title_slide(
        prs,
        "Thank You",
        "Sai Manas Reddy Kayathi\nkmsreddy348@gmail.com\n\n"
        "Ready for questions and live extensions."
    )

    # Save
    output_path = Path("/Users/kingmanas/Desktop/Clara_Presentation.pptx")
    prs.save(str(output_path))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    build()
