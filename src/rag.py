"""RAG/Lookup system for retrieving patient data that doesn't fit in the compact context.

Uses LOINC codes for labs, SNOMED codes for conditions, and RxNorm for medications.
This is a code-based lookup (dictionary), not a vector database, because FHIR data
is already coded with standard terminologies — exact retrieval is more reliable
and requires zero infrastructure.
"""

from collections import defaultdict
from src.fhir_parser import extract_observations


class PatientIndex:
    """Searchable index over a single patient's FHIR resources."""

    def __init__(self, resources: dict):
        self.resources = resources
        self._obs_by_code: dict[str, list[dict]] = {}
        self._obs_by_name: dict[str, list[dict]] = {}
        self._build_index()

    def _build_index(self):
        """Build lookup indices from FHIR resources."""
        observations = extract_observations(self.resources)

        for name, readings in observations.items():
            # Index by display name (case-insensitive)
            self._obs_by_name[name.lower()] = readings

            # Index by LOINC code
            if readings and readings[0].get("code"):
                self._obs_by_code[readings[0]["code"]] = readings

    def lookup_lab(self, query: str) -> list[dict]:
        """Look up lab results by LOINC code or name.

        Args:
            query: A LOINC code (e.g., "4548-4") or a partial name
                   (e.g., "hba1c", "hemoglobin a1c").

        Returns:
            List of observations sorted by date, or empty list.
        """
        # Try exact LOINC code match
        if query in self._obs_by_code:
            return self._obs_by_code[query]

        # Try name match (case-insensitive, partial)
        query_lower = query.lower()
        for name, readings in self._obs_by_name.items():
            if query_lower in name:
                return readings

        return []

    def lookup_all_lab_codes(self) -> list[dict]:
        """Return all available lab test types with their codes and counts."""
        results = []
        for name, readings in sorted(self._obs_by_name.items()):
            code = readings[0].get("code", "") if readings else ""
            latest = readings[-1] if readings else {}
            results.append({
                "name": name,
                "code": code,
                "count": len(readings),
                "latest_date": latest.get("date", "")[:10],
                "latest_value": latest.get("value"),
                "unit": latest.get("unit", ""),
            })
        return results

    def search(self, query: str) -> dict:
        """General search across all indexed data.

        Returns a dict with matching results from different categories.
        """
        results = {"labs": [], "conditions": [], "medications": []}

        # Search labs
        lab_results = self.lookup_lab(query)
        if lab_results:
            results["labs"] = lab_results

        # Search conditions
        query_lower = query.lower()
        for cond in self.resources.get("Condition", []):
            coding = cond.get("code", {}).get("coding", [{}])
            display = coding[0].get("display", "") if coding else ""
            if query_lower in display.lower():
                results["conditions"].append({
                    "display": display,
                    "onset": cond.get("onsetDateTime", "")[:10],
                    "status": cond.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", ""),
                })

        # Search medications
        for med in self.resources.get("MedicationRequest", []):
            coding = med.get("medicationCodeableConcept", {}).get("coding", [{}])
            display = coding[0].get("display", "") if coding else ""
            if query_lower in display.lower():
                results["medications"].append({
                    "display": display,
                    "status": med.get("status", ""),
                    "date": med.get("authoredOn", "")[:10],
                })

        return results
