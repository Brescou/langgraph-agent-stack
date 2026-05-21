"""
tests/test_pack_contracts.py — Contract tests for BaseDomainPack and domain packs.

Verifies:
- BaseDomainPack exposes default input/output schemas as ClassVars.
- Concrete packs can override schemas with typed Pydantic models.
- Schema validation works correctly (accepts valid input, rejects invalid).
- Helper constructors (e.g. from_analysis_report) produce well-formed outputs.
- PackRegistry registration, retrieval, and BaseDomainPack contract (methods, attrs).
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from agents.models import AnalysisReport, ResearchResult
from domain_packs.analysis_only.pack import AnalysisOnlyPack
from domain_packs.analysis_only.schemas import AnalysisOnlyInput, AnalysisOnlyOutput
from domain_packs.research_analysis.pack import ResearchAnalysisPack
from domain_packs.research_analysis.schemas import (
    ResearchAnalysisInput,
    ResearchAnalysisOutput,
)
from domain_packs.research_only.pack import ResearchOnlyPack
from domain_packs.research_only.schemas import ResearchOnlyInput, ResearchOnlyOutput
from domain_packs.summariser.pack import SummariserPack
from domain_packs.summariser.schemas import SummaryInput, SummaryOutput
from pack_kernel.base_pack import (
    BaseDomainPack,
    _DefaultPackInput,
    _DefaultPackOutput,
    normalize_pack_stream_event,
    pack_stream_event,
)
from pack_kernel.registry import PackRegistry

# ---------------------------------------------------------------------------
# BaseDomainPack default schema tests
# ---------------------------------------------------------------------------


def test_base_pack_has_default_input_schema() -> None:
    """BaseDomainPack.input_schema must equal _DefaultPackInput."""
    assert BaseDomainPack.input_schema is _DefaultPackInput


def test_base_pack_has_default_output_schema() -> None:
    """BaseDomainPack.output_schema must equal _DefaultPackOutput."""
    assert BaseDomainPack.output_schema is _DefaultPackOutput


def test_pack_stream_event_uses_top_level_type_field() -> None:
    event = pack_stream_event("phase_started", phase="research")
    assert event == {"type": "phase_started", "phase": "research"}


def test_normalize_pack_stream_event_converts_legacy_shape() -> None:
    legacy = {"event": "phase_started", "data": {"phase": "research"}}
    assert normalize_pack_stream_event(legacy) == {
        "type": "phase_started",
        "phase": "research",
    }


# ---------------------------------------------------------------------------
# ResearchAnalysisPack custom schema declarations
# ---------------------------------------------------------------------------


def test_research_pack_has_custom_input_schema() -> None:
    """ResearchAnalysisPack.input_schema must be ResearchAnalysisInput (not the default)."""
    assert ResearchAnalysisPack.input_schema is ResearchAnalysisInput


def test_research_pack_has_custom_output_schema() -> None:
    """ResearchAnalysisPack.output_schema must be ResearchAnalysisOutput (not the default)."""
    assert ResearchAnalysisPack.output_schema is ResearchAnalysisOutput


# ---------------------------------------------------------------------------
# ResearchAnalysisInput validation
# ---------------------------------------------------------------------------


def test_research_pack_input_schema_validates_query() -> None:
    """ResearchAnalysisInput accepts a non-empty query with default max_sources."""
    inp = ResearchAnalysisInput(query="test")
    assert inp.query == "test"
    assert inp.max_sources == 5


def test_research_pack_input_schema_rejects_empty_query() -> None:
    """ResearchAnalysisInput must reject an empty query string."""
    with pytest.raises(ValidationError):
        ResearchAnalysisInput(query="")


def test_research_pack_input_schema_accepts_custom_max_sources() -> None:
    """ResearchAnalysisInput accepts a valid max_sources within bounds."""
    inp = ResearchAnalysisInput(query="hello", max_sources=20)
    assert inp.max_sources == 20


def test_research_pack_input_schema_rejects_max_sources_zero() -> None:
    """ResearchAnalysisInput must reject max_sources=0 (ge=1 constraint)."""
    with pytest.raises(ValidationError):
        ResearchAnalysisInput(query="hello", max_sources=0)


def test_research_pack_input_schema_rejects_max_sources_over_limit() -> None:
    """ResearchAnalysisInput must reject max_sources > 50 (le=50 constraint)."""
    with pytest.raises(ValidationError):
        ResearchAnalysisInput(query="hello", max_sources=51)


# ---------------------------------------------------------------------------
# ResearchAnalysisOutput.from_analysis_report
# ---------------------------------------------------------------------------


def test_research_pack_output_schema_from_analysis_report() -> None:
    """from_analysis_report must map all AnalysisReport fields onto the output schema."""
    report = AnalysisReport(
        query="What is quantum computing?",
        executive_summary="A paradigm shift in computational power.",
        key_insights=["Qubits enable superposition."],
        patterns=["Rapid hardware iteration."],
        implications=["Cryptographic systems need replacement."],
        confidence=0.82,
        research_summary="Quantum computing uses qubits.",
        metadata={"run_id": "test-run-001"},
    )

    output = ResearchAnalysisOutput.from_analysis_report(report, cost_usd=0.05)

    assert output.query == report.query
    assert output.executive_summary == report.executive_summary
    assert output.key_insights == report.key_insights
    assert output.patterns == report.patterns
    assert output.implications == report.implications
    assert output.confidence == report.confidence
    assert output.research_summary == report.research_summary
    assert output.cost_usd == pytest.approx(0.05)


def test_research_pack_output_schema_from_analysis_report_no_cost() -> None:
    """from_analysis_report with no cost_usd must default to None."""
    report = AnalysisReport(
        query="test",
        executive_summary="summary",
        key_insights=[],
        patterns=[],
        implications=[],
        confidence=0.5,
        research_summary="research",
    )

    output = ResearchAnalysisOutput.from_analysis_report(report)
    assert output.cost_usd is None


def test_research_pack_output_schema_from_analysis_report_rejects_wrong_type() -> None:
    """from_analysis_report must raise TypeError when passed a non-AnalysisReport."""
    with pytest.raises(TypeError):
        ResearchAnalysisOutput.from_analysis_report({"query": "bad"})


# ---------------------------------------------------------------------------
# PackRegistry.get_schemas
# ---------------------------------------------------------------------------


def test_get_schemas_returns_correct_tuple() -> None:
    """get_schemas('research_analysis') must return (ResearchAnalysisInput, ResearchAnalysisOutput)."""
    input_schema, output_schema = PackRegistry.get_schemas("research_analysis")
    assert input_schema is ResearchAnalysisInput
    assert output_schema is ResearchAnalysisOutput


def test_get_schemas_raises_for_unknown_pack() -> None:
    """get_schemas must raise KeyError for an unregistered pack_id."""
    with pytest.raises(KeyError):
        PackRegistry.get_schemas("nonexistent")


# ---------------------------------------------------------------------------
# PackRegistry.list_packs_with_metadata
# ---------------------------------------------------------------------------


def test_list_packs_with_metadata_structure() -> None:
    """Every item returned by list_packs_with_metadata must have the five required keys."""
    metadata = PackRegistry.list_packs_with_metadata()
    assert len(metadata) >= 1, "At least one pack must be registered"
    for item in metadata:
        assert "pack_id" in item
        assert "name" in item
        assert "description" in item
        assert "input_schema" in item
        assert "output_schema" in item


def test_list_packs_with_metadata_includes_json_schema() -> None:
    """input_schema and output_schema in metadata must be valid JSON Schema dicts."""
    metadata = PackRegistry.list_packs_with_metadata()
    assert len(metadata) >= 1, "At least one pack must be registered"
    for item in metadata:
        input_js = item["input_schema"]
        output_js = item["output_schema"]
        assert isinstance(input_js, dict)
        assert isinstance(output_js, dict)
        # JSON Schema standard: top-level dict must contain 'type' or 'properties'
        assert "type" in input_js or "properties" in input_js, (
            f"input_schema for {item['pack_id']} is not a valid JSON Schema dict: {input_js}"
        )
        assert "type" in output_js or "properties" in output_js, (
            f"output_schema for {item['pack_id']} is not a valid JSON Schema dict: {output_js}"
        )


# ---------------------------------------------------------------------------
# PackRegistry — registration and retrieval
# ---------------------------------------------------------------------------


def test_research_analysis_pack_is_registered() -> None:
    """``research_analysis`` must appear in ``list_packs()``."""
    assert "research_analysis" in PackRegistry.list_packs()


def test_research_only_pack_is_registered() -> None:
    """``research_only`` must appear in ``list_packs()``."""
    assert "research_only" in PackRegistry.list_packs()


def test_summariser_pack_is_registered() -> None:
    """``summariser`` must appear in ``list_packs()``."""
    assert "summariser" in PackRegistry.list_packs()


def test_analysis_only_pack_is_registered() -> None:
    """``analysis_only`` must appear in ``list_packs()``."""
    assert "analysis_only" in PackRegistry.list_packs()


def test_registry_get_summariser_returns_correct_class() -> None:
    assert PackRegistry.get("summariser") is SummariserPack


def test_registry_get_analysis_only_returns_correct_class() -> None:
    assert PackRegistry.get("analysis_only") is AnalysisOnlyPack


def test_registry_get_research_only_returns_correct_class() -> None:
    """``get('research_only')`` must return ``ResearchOnlyPack``."""
    assert PackRegistry.get("research_only") is ResearchOnlyPack


def test_registry_get_returns_correct_class() -> None:
    """``get('research_analysis')`` must return ``ResearchAnalysisPack``."""
    resolved = PackRegistry.get("research_analysis")
    assert resolved is ResearchAnalysisPack


def test_registry_get_unknown_raises_key_error() -> None:
    """Unregistered pack_id must raise KeyError."""
    with pytest.raises(KeyError, match="not registered"):
        PackRegistry.get("nonexistent_pack_xyz")


def test_registry_list_packs_returns_sorted_list() -> None:
    """Pack IDs must be returned sorted."""
    packs = PackRegistry.list_packs()
    assert isinstance(packs, list)
    assert packs == sorted(packs)


def test_registry_register_without_pack_id_raises() -> None:
    """Registering a class without ``pack_id`` must raise ValueError."""

    class BadPack(BaseDomainPack):
        pass

    with pytest.raises(ValueError, match="pack_id"):
        PackRegistry.register(BadPack)


# ---------------------------------------------------------------------------
# BaseDomainPack — abstract contract on ResearchAnalysisPack
# ---------------------------------------------------------------------------


def test_research_analysis_pack_inherits_base() -> None:
    assert issubclass(ResearchAnalysisPack, BaseDomainPack)


def test_research_analysis_pack_has_required_class_attrs() -> None:
    assert ResearchAnalysisPack.pack_id == "research_analysis"
    assert isinstance(ResearchAnalysisPack.name, str) and ResearchAnalysisPack.name
    assert (
        isinstance(ResearchAnalysisPack.description, str)
        and ResearchAnalysisPack.description
    )


def test_research_analysis_pack_implements_abstract_methods() -> None:
    abstract_methods = {"run", "arun", "_iter_stream_events"}
    pack_methods = set(dir(ResearchAnalysisPack))
    missing = abstract_methods - pack_methods
    assert not missing, f"Missing abstract method implementations: {missing}"


def test_research_analysis_pack_exposes_normalized_stream_events() -> None:
    assert inspect.isasyncgenfunction(ResearchAnalysisPack.stream_events)
    assert inspect.isasyncgenfunction(ResearchAnalysisPack._iter_stream_events)


def test_research_analysis_pack_run_is_not_abstract() -> None:
    assert not getattr(ResearchAnalysisPack.run, "__isabstractmethod__", False)


def test_research_analysis_pack_arun_is_coroutine() -> None:
    assert inspect.iscoroutinefunction(ResearchAnalysisPack.arun)


def test_research_analysis_pack_supports_context_manager() -> None:
    assert hasattr(ResearchAnalysisPack, "__enter__")
    assert hasattr(ResearchAnalysisPack, "__exit__")


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


def test_default_pack_id_resolves_to_registered_pack() -> None:
    """``DEFAULT_PACK_ID`` from settings must map to a registered pack class."""
    from core.config import get_settings

    settings = get_settings()
    pack_id = settings.default_pack_id
    resolved = PackRegistry.get(pack_id)
    assert issubclass(resolved, BaseDomainPack)


def test_research_only_pack_schemas() -> None:
    """ResearchOnlyPack must declare typed input/output schemas."""
    assert ResearchOnlyPack.input_schema is ResearchOnlyInput
    assert ResearchOnlyPack.output_schema is ResearchOnlyOutput


def test_research_only_output_from_research_result() -> None:
    """from_research_result must map ResearchResult fields."""
    result = ResearchResult(
        query="demo topic",
        findings=["a"],
        summary="s",
        sources=["https://example.com"],
        confidence=0.9,
    )
    output = ResearchOnlyOutput.from_research_result(result, cost_usd=0.01)
    assert output.query == "demo topic"
    assert output.findings == ["a"]
    assert output.cost_usd == 0.01


def test_summariser_pack_schemas() -> None:
    assert SummariserPack.input_schema is SummaryInput
    assert SummariserPack.output_schema is SummaryOutput


def test_analysis_only_pack_schemas() -> None:
    assert AnalysisOnlyPack.input_schema is AnalysisOnlyInput
    assert AnalysisOnlyPack.output_schema is AnalysisOnlyOutput


def test_summary_input_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        SummaryInput(text="")


def test_analysis_only_input_requires_query() -> None:
    with pytest.raises(ValidationError):
        AnalysisOnlyInput(query="")
