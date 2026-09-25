"""Tests for the compaction pipeline."""

import json
from pathlib import Path

import pytest

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
from src.evals import run_generic_evals, run_merlene_evals

DATA_DIR = Path("data")


def get_merlene_path():
    paths = list(DATA_DIR.glob("Merlene*.json"))
    if not paths:
        pytest.skip("Data not extracted — run: tar -xJf data/synthea_sample_data_fhir_latest.tar.xz -C data --strip-components=1")
    return paths[0]


def get_any_patient_path():
    paths = [
        p for p in DATA_DIR.glob("*.json")
        if not p.name.startswith("hospital") and not p.name.startswith("practitioner")
    ]
    if not paths:
        pytest.skip("Data not extracted")
    return paths[0]


class TestFHIRParser:
    def test_load_and_parse(self):
        path = get_any_patient_path()
        bundle = load_bundle(path)
        parsed = parse_bundle(bundle)
        assert "resources" in parsed
        assert "Patient" in parsed["resources"]

    def test_demographics(self):
        path = get_merlene_path()
        bundle = load_bundle(path)
        resources = parse_bundle(bundle)["resources"]
        demo = extract_patient_demographics(resources)
        assert "Merlene" in demo["name"]
        assert demo["gender"] == "female"

    def test_merlene_allergies(self):
        path = get_merlene_path()
        bundle = load_bundle(path)
        resources = parse_bundle(bundle)["resources"]
        allergies = extract_allergies(resources)
        assert len(allergies) == 0, "Merlene should have zero allergies"

    def test_merlene_hba1c(self):
        path = get_merlene_path()
        bundle = load_bundle(path)
        resources = parse_bundle(bundle)["resources"]
        observations = extract_observations(resources)

        hba1c_key = None
        for key in observations:
            if "a1c" in key.lower():
                hba1c_key = key
                break

        assert hba1c_key is not None, "Should find HbA1c observations"
        readings = observations[hba1c_key]
        assert len(readings) == 8, f"Expected 8 HbA1c readings, got {len(readings)}"

        latest = readings[-1]
        assert latest["value"] == 6.31
        assert "2025-09-29" in latest["date"]

    def test_merlene_conditions(self):
        path = get_merlene_path()
        bundle = load_bundle(path)
        resources = parse_bundle(bundle)["resources"]
        active, resolved = extract_conditions(resources)

        prediabetes = [c for c in active if "prediabet" in c["display"].lower()]
        assert len(prediabetes) > 0, "Merlene should have active prediabetes"


class TestCompactor:
    def test_compact_produces_output(self):
        path = get_any_patient_path()
        bundle = load_bundle(path)
        resources = parse_bundle(bundle)["resources"]
        context = compact_bundle(resources)
        assert len(context) > 100
        assert "# Patient:" in context

    def test_compact_has_required_sections(self):
        path = get_merlene_path()
        bundle = load_bundle(path)
        resources = parse_bundle(bundle)["resources"]
        context = compact_bundle(resources)

        assert "## Allergies" in context
        assert "## Active Problems" in context
        assert "## Medications" in context
        assert "No allergies recorded" in context

    def test_compact_under_900k(self):
        path = get_merlene_path()
        bundle = load_bundle(path)
        resources = parse_bundle(bundle)["resources"]
        context = compact_bundle(resources)
        tokens = len(context) // 4  # chars/4 estimate
        assert tokens <= 900_000


class TestRAG:
    def test_lookup_by_name(self):
        path = get_merlene_path()
        bundle = load_bundle(path)
        resources = parse_bundle(bundle)["resources"]
        index = PatientIndex(resources)

        results = index.lookup_lab("a1c")
        assert len(results) == 8

    def test_lookup_all_codes(self):
        path = get_merlene_path()
        bundle = load_bundle(path)
        resources = parse_bundle(bundle)["resources"]
        index = PatientIndex(resources)

        codes = index.lookup_all_lab_codes()
        assert len(codes) > 0


class TestEvals:
    def test_merlene_evals_pass(self):
        path = get_merlene_path()
        bundle = load_bundle(path)
        resources = parse_bundle(bundle)["resources"]
        context = compact_bundle(resources)
        observations = extract_observations(resources)

        suite = run_merlene_evals(context, observations)
        assert suite.pass_rate >= 0.8, f"Expected >=80% pass rate, got {suite.pass_rate:.0%}: {suite.summary()}"

    def test_generic_evals_pass(self):
        path = get_any_patient_path()
        bundle = load_bundle(path)
        resources = parse_bundle(bundle)["resources"]
        context = compact_bundle(resources)

        suite = run_generic_evals(context, resources)
        assert suite.pass_rate == 1.0, f"All generic evals should pass: {suite.summary()}"
