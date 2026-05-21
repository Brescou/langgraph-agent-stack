"""tests/test_summariser_pack.py — Unit tests for SummariserPack."""

from __future__ import annotations

from unittest.mock import MagicMock

from domain_packs.summariser.pack import SummariserPack
from domain_packs.summariser.schemas import SummaryInput, SummaryOutput


def test_summariser_run_from_input_returns_summary_output() -> None:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="- First point\n- Second point\n- Third point"
    )
    pack = SummariserPack(run_id="sum-test-1", llm=mock_llm)
    result = pack.run_from_input(
        SummaryInput(text="Some long article about AI.", bullet_count=3)
    )

    assert isinstance(result, SummaryOutput)
    assert result.original_length == len("Some long article about AI.")
    assert len(result.bullets) == 3
    mock_llm.invoke.assert_called_once()


def test_summary_output_from_summary_result() -> None:
    inner = SummaryOutput(original_length=10, bullets=["a", "b"])
    output = SummaryOutput.from_summary_result(inner, cost_usd=0.02)
    assert output.bullets == ["a", "b"]
    assert output.cost_usd == 0.02
