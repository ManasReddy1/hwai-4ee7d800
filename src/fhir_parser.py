"""Parse FHIR bundles into structured Python objects."""

import json
import base64
from collections import defaultdict
from pathlib import Path


def load_bundle(path: Path) -> dict:
    """Load a FHIR Bundle JSON file."""
    with open(path) as f:
        return json.load(f)


def parse_bundle(bundle: dict) -> dict:
    """Parse a FHIR Bundle into categorized resources.

    Returns a dict keyed by resourceType, each value is a list of resources.
    Also builds a fullUrl -> resource lookup for resolving internal references.
    """
    resources_by_type = defaultdict(list)
    url_lookup = {}

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType", "Unknown")
        resources_by_type[rtype].append(resource)

        full_url = entry.get("fullUrl", "")
        if full_url:
            url_lookup[full_url] = resource

    return {
        "resources": dict(resources_by_type),
        "url_lookup": url_lookup,
    }


def extract_patient_demographics(resources: dict) -> dict:
    """Extract patient name, DOB, gender, deceased status."""
    patients = resources.get("Patient", [])
    if not patients:
        return {}

    pt = patients[0]
    names = pt.get("name", [{}])
    name_parts = names[0] if names else {}
    given = " ".join(name_parts.get("given", []))
    family = name_parts.get("family", "")

    return {
        "name": f"{given} {family}".strip(),
        "birth_date": pt.get("birthDate", "Unknown"),
        "gender": pt.get("gender", "Unknown"),
        "deceased_date": pt.get("deceasedDateTime"),
        "address": _extract_address(pt),
    }


def _extract_address(pt: dict) -> str:
    addrs = pt.get("address", [])
    if not addrs:
        return ""
    a = addrs[0]
    city = a.get("city", "")
    state = a.get("state", "")
    return f"{city}, {state}" if city else ""


def extract_conditions(resources: dict) -> tuple[list[dict], list[dict]]:
    """Extract active and resolved conditions.

    Returns (active_conditions, resolved_conditions).
    """
    active = []
    resolved = []

    for cond in resources.get("Condition", []):
        coding = cond.get("code", {}).get("coding", [{}])
        display = coding[0].get("display", "Unknown") if coding else "Unknown"
        code = coding[0].get("code", "") if coding else ""
        onset = cond.get("onsetDateTime", "")[:10]

        status_coding = cond.get("clinicalStatus", {}).get("coding", [{}])
        status = status_coding[0].get("code", "") if status_coding else ""

        abatement = cond.get("abatementDateTime", "")[:10]

        entry = {
            "display": display,
            "code": code,
            "onset": onset,
            "status": status,
            "abatement": abatement,
        }

        if status == "resolved":
            resolved.append(entry)
        else:
            active.append(entry)

    return active, resolved


def extract_allergies(resources: dict) -> list[dict]:
    """Extract allergy information."""
    allergies = []
    for allergy in resources.get("AllergyIntolerance", []):
        coding = allergy.get("code", {}).get("coding", [{}])
        display = coding[0].get("display", "Unknown") if coding else "Unknown"

        reactions = []
        for reaction in allergy.get("reaction", []):
            for manifestation in reaction.get("manifestation", []):
                r_coding = manifestation.get("coding", [{}])
                if r_coding:
                    reactions.append(r_coding[0].get("display", ""))

        allergies.append({
            "substance": display,
            "reactions": reactions,
        })
    return allergies


def extract_medications(resources: dict) -> list[dict]:
    """Extract deduplicated medications."""
    seen = {}
    for med in resources.get("MedicationRequest", []):
        coding = med.get("medicationCodeableConcept", {}).get("coding", [{}])
        display = coding[0].get("display", "Unknown") if coding else "Unknown"
        code = coding[0].get("code", "") if coding else ""
        status = med.get("status", "unknown")
        authored = med.get("authoredOn", "")[:10]

        # Keep the most recent entry for each medication
        if display not in seen or authored > seen[display]["authored"]:
            seen[display] = {
                "display": display,
                "code": code,
                "status": status,
                "authored": authored,
            }

    return sorted(seen.values(), key=lambda m: m["authored"], reverse=True)


def extract_observations(resources: dict) -> dict[str, list[dict]]:
    """Extract observations grouped by LOINC code.

    Returns dict: display_name -> list of {date, value, unit, code}.
    """
    obs_by_code = defaultdict(list)

    for obs in resources.get("Observation", []):
        coding = obs.get("code", {}).get("coding", [{}])
        if not coding:
            continue
        display = coding[0].get("display", coding[0].get("code", "unknown"))
        code = coding[0].get("code", "")

        value_qty = obs.get("valueQuantity", {})
        value = value_qty.get("value")
        unit = value_qty.get("unit", "")

        # Handle coded values (e.g., "Appearance of Urine: Yellow")
        if value is None:
            value_cc = obs.get("valueCodeableConcept", {}).get("coding", [{}])
            if value_cc:
                value = value_cc[0].get("display", "")
                unit = ""

        date = obs.get("effectiveDateTime", obs.get("issued", ""))

        obs_by_code[display].append({
            "date": date,
            "value": value,
            "unit": unit,
            "code": code,
        })

    # Sort each group by date
    for key in obs_by_code:
        obs_by_code[key].sort(key=lambda x: x["date"])

    return dict(obs_by_code)


def extract_clinical_notes(resources: dict) -> list[dict]:
    """Extract clinical notes from DocumentReference only (not DiagnosticReport
    since they are 100% duplicates)."""
    notes = []

    for doc_ref in resources.get("DocumentReference", []):
        date = doc_ref.get("date", "")[:10]
        for content in doc_ref.get("content", []):
            att = content.get("attachment", {})
            if att.get("data"):
                try:
                    text = base64.b64decode(att["data"]).decode(
                        "utf-8", errors="replace"
                    )
                    notes.append({"date": date, "text": text})
                except Exception:
                    pass

    notes.sort(key=lambda n: n["date"], reverse=True)
    return notes


def extract_immunizations(resources: dict) -> list[dict]:
    """Extract immunization records."""
    immunizations = []
    for imm in resources.get("Immunization", []):
        coding = imm.get("vaccineCode", {}).get("coding", [{}])
        display = coding[0].get("display", "Unknown") if coding else "Unknown"
        date = imm.get("occurrenceDateTime", "")[:10]
        immunizations.append({"vaccine": display, "date": date})

    immunizations.sort(key=lambda i: i["date"], reverse=True)
    return immunizations


def extract_procedures(resources: dict) -> list[dict]:
    """Extract procedures."""
    procedures = []
    for proc in resources.get("Procedure", []):
        coding = proc.get("code", {}).get("coding", [{}])
        display = coding[0].get("display", "Unknown") if coding else "Unknown"
        date = proc.get("performedDateTime", proc.get("performedPeriod", {}).get("start", ""))
        if date:
            date = date[:10]
        status = proc.get("status", "")
        procedures.append({"display": display, "date": date, "status": status})

    procedures.sort(key=lambda p: p["date"] or "", reverse=True)
    return procedures


def extract_encounters(resources: dict) -> list[dict]:
    """Extract encounter summary info."""
    encounters = []
    for enc in resources.get("Encounter", []):
        period = enc.get("period", {})
        start = period.get("start", "")
        enc_type = enc.get("type", [{}])
        type_display = enc_type[0].get("text", "") if enc_type else ""
        encounters.append({"start": start, "type": type_display})

    encounters.sort(key=lambda e: e["start"], reverse=True)
    return encounters
