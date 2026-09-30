"""Synthetic tests for schema-eligible automated refusal signals."""

import json

import pytest

from evaluation.metrics import automated_refusal_signal
from evaluation.schemas import BenchmarkRecord, GenerationObservation, OperationalStatus


def _record(answerability="unanswerable", question_id="fictional-item"):
    unanswerable = answerability == "unanswerable"
    return BenchmarkRecord.model_validate_json(json.dumps({
        "id": question_id, "benchmark_id": "fictional-lab", "benchmark_version": "1.0.0",
        "review_status": "reviewed",
        "category": answerability if answerability != "answerable" else "direct_factual",
        "question": "Which fictional item is described?",
        "answerability": answerability,
        "expected_behavior": "refuse_or_state_insufficient_information" if unanswerable
            else "clarify_or_qualify" if answerability == "ambiguous" else "answer",
        "reference_answer": None if unanswerable else "A fictional item.",
        "expected_points": [{"id": "manual", "text": "Explain availability.", "automated_match": False}],
        "evidence": [] if unanswerable else [{
            "evidence_id": "fictional-evidence", "pdf_page": 1, "printed_page_label": None,
            "section": None, "anchor": "Fictional item.", "notes": None,
        }],
        "notes": None,
    }))


def _generation(answer, status=OperationalStatus.SUCCESS):
    return GenerationObservation(
        status=status, latency_ms=1.0 if status is OperationalStatus.SUCCESS else None,
        generation_model="synthetic", answer=answer,
        context_available=False if status is OperationalStatus.SUCCESS else None,
        sources=(), citation_validation=None, error=None,
    )


@pytest.mark.parametrize("answer", [
    "The corpus does not provide the requested fictional detail.",
    "That detail is not specified in the corpus.",
    "The detail cannot be determined from the corpus.",
    "There is insufficient information in the corpus.",
    "The corpus is missing that detail, so I will not invent it.",
])
def test_refusal_signal_positive_patterns(answer):
    assert automated_refusal_signal(_record(), _generation(answer)) is True


@pytest.mark.parametrize("question_id", ["fictional-a", "fictional-b", "fictional-c"])
def test_eligibility_does_not_depend_on_benchmark_ids(question_id):
    assert automated_refusal_signal(
        _record(question_id=question_id), _generation("The corpus does not provide that detail."),
    ) is True


@pytest.mark.parametrize("answer", [
    "I don't know.", "Please ask elsewhere.",
    "The fictional item is violet.", "The corpus describes fictional objects.",
])
def test_refusal_signal_negative_patterns(answer):
    assert automated_refusal_signal(_record(), _generation(answer)) is False


@pytest.mark.parametrize("answerability", ["answerable", "ambiguous"])
def test_non_unanswerable_questions_are_ineligible(answerability):
    assert automated_refusal_signal(
        _record(answerability), _generation("The corpus does not provide that detail."),
    ) is None


@pytest.mark.parametrize("generation", [
    None, _generation(None),
    _generation("The corpus does not state it.", OperationalStatus.GENERATION_FAILURE),
])
def test_generation_unavailable_or_failed_is_null(generation):
    assert automated_refusal_signal(_record(), generation) is None


def test_signal_does_not_validate_later_claims():
    answer = "The corpus does not provide that detail. An unsupported claim follows."
    assert automated_refusal_signal(_record(), _generation(answer)) is True
