"""
tests/test_reviews.py — Human-review queue for regulated pack outputs.

Covers:
- ``core/review_store.py`` backends (in-memory and SQLite): create / get /
  list with status filter and pagination / decide, duplicate creation,
  conflict on double decision, SQLite persistence across reopen;
- the ``/reviews`` API endpoints (list, detail, decision, 404/409);
- automatic creation of a pending review after a regulated pack run, and
  the best-effort guarantee (a broken store never fails the run).

All agent calls are mocked; no real LLM requests are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api.state as state
from core.review_store import (
    InMemoryReviewStore,
    ReviewAlreadyDecidedError,
    ReviewNotFoundError,
    SqliteReviewStore,
    create_review_store,
    summarize_output,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_review_store():
    """Swap in a fresh in-memory review store before and after each test."""
    state.review_store = InMemoryReviewStore()
    yield
    state.review_store = InMemoryReviewStore()


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    """Yield each review-store backend, closing it afterwards."""
    if request.param == "memory":
        backend = InMemoryReviewStore()
    else:
        backend = SqliteReviewStore(db_path=str(tmp_path / "reviews.db"))
    yield backend
    backend.close()


# ---------------------------------------------------------------------------
# Store backends (unit)
# ---------------------------------------------------------------------------


class TestReviewStoreBackends:
    def test_create_then_get(self, store) -> None:
        record = store.create(
            run_id="run-1",
            pack_id="talent_screening",
            session_id="sess-1",
            output_summary="fit_score 0.8",
        )
        assert record.status == "pending"
        assert record.created_at

        fetched = store.get("run-1")
        assert fetched is not None
        assert fetched.pack_id == "talent_screening"
        assert fetched.session_id == "sess-1"
        assert fetched.output_summary == "fit_score 0.8"

    def test_get_unknown_returns_none(self, store) -> None:
        assert store.get("run-unknown") is None

    def test_duplicate_create_raises(self, store) -> None:
        store.create(run_id="run-dup", pack_id="financial_memo")
        with pytest.raises(ValueError, match="exists"):
            store.create(run_id="run-dup", pack_id="financial_memo")

    def test_list_filters_by_status_and_paginates(self, store) -> None:
        for i in range(5):
            store.create(run_id=f"run-{i}", pack_id="contract_reviewer")
        store.decide("run-0", status="approved", reviewer="alice")

        pending = store.list_reviews(status="pending")
        assert [r.run_id for r in pending] == ["run-4", "run-3", "run-2", "run-1"]

        page = store.list_reviews(status="pending", limit=2, offset=1)
        assert [r.run_id for r in page] == ["run-3", "run-2"]

        approved = store.list_reviews(status="approved")
        assert [r.run_id for r in approved] == ["run-0"]
        assert approved[0].reviewer == "alice"

    def test_decide_records_decision(self, store) -> None:
        store.create(run_id="run-d", pack_id="hr_policy_qa")
        decided = store.decide(
            run_id="run-d", status="rejected", reviewer="bob", notes="not compliant"
        )
        assert decided.status == "rejected"
        assert decided.reviewer == "bob"
        assert decided.notes == "not compliant"
        assert decided.decided_at is not None

    def test_double_decision_conflicts(self, store) -> None:
        store.create(run_id="run-c", pack_id="financial_memo")
        store.decide(run_id="run-c", status="approved", reviewer="alice")
        with pytest.raises(ReviewAlreadyDecidedError):
            store.decide(run_id="run-c", status="rejected", reviewer="bob")

    def test_decide_unknown_raises_not_found(self, store) -> None:
        with pytest.raises(ReviewNotFoundError):
            store.decide(run_id="run-nope", status="approved", reviewer="alice")

    def test_output_summary_is_truncated(self, store) -> None:
        record = store.create(
            run_id="run-t", pack_id="financial_memo", output_summary="x" * 10_000
        )
        assert len(record.output_summary) == 500


def test_sqlite_store_persists_across_reopen(tmp_path) -> None:
    path = str(tmp_path / "reviews.db")
    first = SqliteReviewStore(db_path=path)
    first.create(run_id="run-p", pack_id="contract_reviewer")
    first.decide(run_id="run-p", status="approved", reviewer="alice")
    first.close()

    reopened = SqliteReviewStore(db_path=path)
    try:
        record = reopened.get("run-p")
        assert record is not None
        assert record.status == "approved"
        assert record.reviewer == "alice"
    finally:
        reopened.close()


def test_factory_builds_each_backend(tmp_path) -> None:
    assert isinstance(create_review_store(backend="memory"), InMemoryReviewStore)
    sqlite_store = create_review_store(
        backend="sqlite", sqlite_path=str(tmp_path / "r.db")
    )
    try:
        assert isinstance(sqlite_store, SqliteReviewStore)
    finally:
        sqlite_store.close()


def test_summarize_output_handles_unserializable() -> None:
    assert summarize_output(None) == ""
    assert summarize_output({"a": 1}) == '{"a": 1}'
    assert "<object" in summarize_output(object())


# ---------------------------------------------------------------------------
# /reviews endpoints
# ---------------------------------------------------------------------------


def test_reviews_endpoints_list_detail_decide(test_client: TestClient) -> None:
    """Full lifecycle through the API: list pending → decide → 409 on retry."""
    state.review_store.create(
        run_id="run-api-1", pack_id="talent_screening", output_summary="s1"
    )
    state.review_store.create(run_id="run-api-2", pack_id="financial_memo")

    listed = test_client.get("/reviews", params={"status": "pending"})
    assert listed.status_code == 200
    body = listed.json()
    assert body["total"] == 2
    assert [r["run_id"] for r in body["reviews"]] == ["run-api-2", "run-api-1"]

    detail = test_client.get("/reviews/run-api-1")
    assert detail.status_code == 200
    assert detail.json()["status"] == "pending"

    decision = test_client.post(
        "/reviews/run-api-1/decision",
        json={"status": "approved", "reviewer": "alice", "notes": "ok"},
    )
    assert decision.status_code == 200
    assert decision.json()["status"] == "approved"
    assert decision.json()["reviewer"] == "alice"

    retry = test_client.post(
        "/reviews/run-api-1/decision",
        json={"status": "rejected", "reviewer": "bob"},
    )
    assert retry.status_code == 409

    assert test_client.get("/reviews/run-api-1").json()["status"] == "approved"


def test_reviews_detail_unknown_returns_404(test_client: TestClient) -> None:
    assert test_client.get("/reviews/run-ghost").status_code == 404


def test_reviews_decision_rejects_extra_fields(test_client: TestClient) -> None:
    """ReviewDecisionRequest has extra='forbid'."""
    state.review_store.create(run_id="run-x", pack_id="financial_memo")
    response = test_client.post(
        "/reviews/run-x/decision",
        json={"status": "approved", "reviewer": "alice", "sneaky": True},
    )
    assert response.status_code == 422


def test_reviews_unavailable_store_returns_503(test_client: TestClient) -> None:
    with patch("api.state.review_store", None):
        assert test_client.get("/reviews").status_code == 503


# ---------------------------------------------------------------------------
# Automatic review creation after regulated pack runs
# ---------------------------------------------------------------------------


def _regulated_settings():
    from core.config import Settings

    return Settings(
        llm_provider="anthropic",
        anthropic_api_key="sk-ant-test123456789012345",
        regulated_packs_enabled=True,
        api_key=None,
    )


def _run_talent_screening(test_client: TestClient) -> None:
    """POST a mocked talent_screening run through the dynamic pack route."""
    from domain_packs.hr.talent_screening.pack import TalentScreeningPack
    from domain_packs.hr.talent_screening.schemas import TalentScreeningOutput

    output = TalentScreeningOutput(
        fit_score=0.8,
        matched_skills=["python"],
        gaps=[],
        interview_questions=["Tell me about a project."],
        red_flags=[],
        summary_for_hiring_manager="Solid profile.",
        confidence=0.7,
        disclaimer="Assistive output — a human makes the hiring decision.",
    )

    def _noop_init(self, **kwargs):  # type: ignore[override]
        pass

    with (
        patch("api.pack_execution.get_settings", return_value=_regulated_settings()),
        patch.object(TalentScreeningPack, "__init__", _noop_init),
        patch.object(TalentScreeningPack, "run_from_input", return_value=output),
        patch.object(TalentScreeningPack, "close", return_value=None),
    ):
        response = test_client.post(
            "/packs/talent_screening/run",
            json={
                "job_description": "Python developer",
                "resume_text": "Experienced Python developer",
            },
        )
    assert response.status_code == 200


def test_regulated_run_queues_pending_review(test_client: TestClient) -> None:
    _run_talent_screening(test_client)

    pending = state.review_store.list_reviews(status="pending")
    assert len(pending) == 1
    assert pending[0].pack_id == "talent_screening"
    assert pending[0].status == "pending"
    assert "fit_score" in pending[0].output_summary


def test_non_regulated_run_creates_no_review(
    test_client: TestClient, mock_analysis_report
) -> None:
    """research_analysis has no human_review_required policy → no queue entry."""
    from domain_packs.research.research_analysis.pack import ResearchAnalysisPack

    def _noop_init(self, **kwargs):  # type: ignore[override]
        pass

    with (
        patch.object(ResearchAnalysisPack, "__init__", _noop_init),
        patch.object(ResearchAnalysisPack, "run", return_value=mock_analysis_report),
        patch.object(ResearchAnalysisPack, "close", return_value=None),
    ):
        response = test_client.post(
            "/packs/research_analysis/run",
            json={"query": "What is a microservice?"},
        )

    assert response.status_code == 200
    assert state.review_store.list_reviews() == []


def test_broken_review_store_does_not_fail_run(test_client: TestClient) -> None:
    """A review store that raises must never fail the regulated run (200)."""
    broken = MagicMock()
    broken.create.side_effect = RuntimeError("review backend down")

    with patch("api.state.review_store", broken):
        _run_talent_screening(test_client)

    broken.create.assert_called_once()
