"""Compact a parsed FHIR bundle into a token-efficient markdown context."""

from src.fhir_parser import (
    extract_allergies,
    extract_clinical_notes,
    extract_conditions,
    extract_encounters,
    extract_immunizations,
    extract_medications,
    extract_observations,
    extract_patient_demographics,
    extract_procedures,
)


def compact_bundle(resources: dict) -> str:
    """Convert parsed FHIR resources into a compact markdown patient summary.

    Follows the IPS (International Patient Summary) section order:
    Demographics -> Allergies -> Problems -> Medications -> Labs -> Notes

    This order matches how clinicians review charts (problem-oriented)
    and places critical info at the start (leveraging the "Lost in the Middle"
    finding that LLMs attend best to the beginning and end of context).
    """
    sections = []

    # --- Demographics ---
    demo = extract_patient_demographics(resources)
    header = f"# Patient: {demo.get('name', 'Unknown')}"
    header += f"\nDOB: {demo.get('birth_date', '?')} | Gender: {demo.get('gender', '?')}"
    if demo.get("address"):
        header += f" | Location: {demo['address']}"
    if demo.get("deceased_date"):
        header += f"\n**Deceased: {demo['deceased_date'][:10]}**"
    sections.append(header)

    # --- Allergies (IPS Required - explicit absence matters) ---
    allergies = extract_allergies(resources)
    sections.append("\n## Allergies")
    if allergies:
        for a in allergies:
            line = f"- {a['substance']}"
            if a["reactions"]:
                line += f" (Reactions: {', '.join(a['reactions'])})"
            sections.append(line)
    else:
        sections.append("No allergies recorded.")

    # --- Active Problems (IPS Required) ---
    active_conds, resolved_conds = extract_conditions(resources)
    sections.append("\n## Active Problems")
    if active_conds:
        for c in sorted(active_conds, key=lambda x: x["onset"], reverse=True):
            sections.append(f"- {c['display']} (onset: {c['onset']})")
    else:
        sections.append("No active problems recorded.")

    # --- Medications (IPS Required) ---
    medications = extract_medications(resources)
    sections.append("\n## Medications")
    if medications:
        active_meds = [m for m in medications if m["status"] == "active"]
        other_meds = [m for m in medications if m["status"] != "active"]

        if active_meds:
            sections.append("### Current")
            for m in active_meds:
                sections.append(f"- {m['display']} (since {m['authored']})")

        if other_meds:
            # Show only unique stopped/completed meds
            sections.append("### Historical")
            for m in other_meds[:20]:  # Cap at 20 to avoid bloat
                sections.append(f"- {m['display']} ({m['status']}, {m['authored']})")
            if len(other_meds) > 20:
                sections.append(f"- ... and {len(other_meds) - 20} more historical medications")
    else:
        sections.append("No medications recorded.")

    # --- Observations (collapsed to latest + trend) ---
    observations = extract_observations(resources)
    sections.append("\n## Lab Results & Observations")
    if observations:
        sections.append("| Test | Latest Value | Date | Trend | Readings |")
        sections.append("|------|-------------|------|-------|----------|")

        for name, readings in sorted(observations.items()):
            latest = readings[-1]
            val = latest["value"]
            if val is None:
                continue

            # Format value
            if isinstance(val, float):
                val_str = f"{val:.2f}" if val != int(val) else str(int(val))
            else:
                val_str = str(val)

            if latest["unit"]:
                val_str += f" {latest['unit']}"

            date_str = latest["date"][:10] if latest["date"] else "?"

            # Compute trend from last 3 numeric readings
            trend = _compute_trend(readings)
            count = len(readings)

            sections.append(
                f"| {name} | {val_str} | {date_str} | {trend} | {count} |"
            )
    else:
        sections.append("No observations recorded.")

    # --- Immunizations ---
    immunizations = extract_immunizations(resources)
    if immunizations:
        sections.append("\n## Immunizations")
        seen = set()
        for imm in immunizations:
            key = imm["vaccine"]
            if key not in seen:
                seen.add(key)
                sections.append(f"- {imm['vaccine']} (latest: {imm['date']})")

    # --- Procedures (recent 5 years) ---
    procedures = extract_procedures(resources)
    if procedures:
        sections.append("\n## Recent Procedures")
        seen = set()
        shown = 0
        for p in procedures:
            key = p["display"]
            if key not in seen and shown < 30:
                seen.add(key)
                sections.append(f"- {p['display']} ({p['date']})")
                shown += 1
        if len(procedures) > shown:
            sections.append(f"- ... and {len(procedures) - shown} more procedures in history")

    # --- Resolved Conditions ---
    if resolved_conds:
        sections.append("\n## Resolved Problems")
        seen = set()
        for c in sorted(resolved_conds, key=lambda x: x["abatement"] or x["onset"], reverse=True):
            if c["display"] not in seen:
                seen.add(c["display"])
                line = f"- {c['display']} (onset: {c['onset']}"
                if c["abatement"]:
                    line += f", resolved: {c['abatement']}"
                line += ")"
                sections.append(line)

    # --- Clinical Notes (deduplicated) ---
    notes = extract_clinical_notes(resources)
    if notes:
        sections.append("\n## Clinical Notes (deduplicated)")
        unique_lines = set()
        for note in notes:
            for line in note["text"].split("\n"):
                stripped = line.strip()
                if stripped:
                    unique_lines.add(stripped)

        # Sort for consistency
        sections.append("\n".join(sorted(unique_lines)))

    # --- Encounters Summary ---
    encounters = extract_encounters(resources)
    if encounters:
        dates = [e["start"][:10] for e in encounters if e["start"]]
        if dates:
            sections.append(
                f"\n## Encounters: {len(encounters)} visits, "
                f"{min(dates)} to {max(dates)}"
            )

    return "\n".join(sections)


def _compute_trend(readings: list[dict]) -> str:
    """Compute a simple trend from the last 3 numeric readings."""
    numeric = [
        r for r in readings
        if isinstance(r["value"], (int, float))
    ]

    if len(numeric) < 2:
        return "-"

    recent = numeric[-3:] if len(numeric) >= 3 else numeric
    values = [r["value"] for r in recent]

    if all(values[i] <= values[i + 1] for i in range(len(values) - 1)):
        if values[-1] > values[0] * 1.05:
            return "rising"
        return "stable"
    elif all(values[i] >= values[i + 1] for i in range(len(values) - 1)):
        if values[-1] < values[0] * 0.95:
            return "falling"
        return "stable"
    else:
        return "variable"
