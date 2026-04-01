"""
tests/test_tools.py — Unit tests for core/tools.py.

Tests cover the mock search tool, the safe calculator tool, and
get_default_tools().  No external API calls are made.
"""

from __future__ import annotations

from core.tools import (
    create_calculator_tool,
    create_search_tool,
    get_default_tools,
)

# ---------------------------------------------------------------------------
# Search tool
# ---------------------------------------------------------------------------


def test_search_tool_mock_returns_string() -> None:
    """
    The mock search tool must return a non-empty string for any query.

    The mock is the default when SEARCH_PROVIDER is not set to a real provider.
    """
    tool = create_search_tool()
    result = tool.invoke("What is quantum computing?")

    assert isinstance(result, str)
    assert len(result) > 0


def test_search_tool_mock_contains_query() -> None:
    """The mock search result must embed the original query in its output."""
    tool = create_search_tool()
    query = "CAP theorem in distributed systems"
    result = tool.invoke(query)

    assert query in result


def test_search_tool_mock_has_source_url() -> None:
    """The mock search result must include at least one source URL."""
    tool = create_search_tool()
    result = tool.invoke("vector databases")

    assert "http" in result


# ---------------------------------------------------------------------------
# Calculator tool
# ---------------------------------------------------------------------------


def test_calculator_addition() -> None:
    """2 + 2 must return '4'."""
    tool = create_calculator_tool()
    result = tool.invoke("2 + 2")

    assert result == "4"


def test_calculator_subtraction() -> None:
    """10 - 3 must return '7'."""
    tool = create_calculator_tool()
    result = tool.invoke("10 - 3")

    assert result == "7"


def test_calculator_multiplication() -> None:
    """3 * 7 must return '21'."""
    tool = create_calculator_tool()
    result = tool.invoke("3 * 7")

    assert result == "21"


def test_calculator_division() -> None:
    """10 / 4 must return '2.5'."""
    tool = create_calculator_tool()
    result = tool.invoke("10 / 4")

    assert result == "2.5"


def test_calculator_integer_division() -> None:
    """10 // 3 must return '3' (floor division)."""
    tool = create_calculator_tool()
    result = tool.invoke("10 // 3")

    assert result == "3"


def test_calculator_modulo() -> None:
    """10 % 3 must return '1'."""
    tool = create_calculator_tool()
    result = tool.invoke("10 % 3")

    assert result == "1"


def test_calculator_power() -> None:
    """2 ** 10 must return '1024'."""
    tool = create_calculator_tool()
    result = tool.invoke("2 ** 10")

    assert result == "1024"


def test_calculator_nested_parentheses() -> None:
    """(3 + 4) * 2 must return '14'."""
    tool = create_calculator_tool()
    result = tool.invoke("(3 + 4) * 2")

    assert result == "14"


def test_calculator_abs_function() -> None:
    """abs(-5) must return '5'."""
    tool = create_calculator_tool()
    result = tool.invoke("abs(-5)")

    assert result == "5"


def test_calculator_round_function() -> None:
    """round(3.7) must return '4'."""
    tool = create_calculator_tool()
    result = tool.invoke("round(3.7)")

    assert result == "4"


def test_calculator_division_by_zero() -> None:
    """
    Division by zero must return an error message string, not crash.

    The tool must handle ZeroDivisionError gracefully and return a
    string prefixed with 'Error:'.
    """
    tool = create_calculator_tool()
    result = tool.invoke("10 / 0")

    assert isinstance(result, str)
    assert result.lower().startswith("error")


def test_calculator_blocks_builtin_import() -> None:
    """
    Attempts to call __import__ or use built-ins outside the whitelist
    must be blocked and return an error string rather than executing code.
    """
    tool = create_calculator_tool()
    result = tool.invoke("__import__('os')")

    assert isinstance(result, str)
    assert result.lower().startswith("error")


def test_calculator_blocks_attribute_access() -> None:
    """
    Attribute access expressions (e.g. os.getcwd) must be blocked.
    """
    tool = create_calculator_tool()
    result = tool.invoke("os.getcwd()")

    assert isinstance(result, str)
    assert result.lower().startswith("error")


def test_calculator_blocks_string_literal() -> None:
    """Non-numeric literals such as strings must be rejected."""
    tool = create_calculator_tool()
    result = tool.invoke('"hello"')

    assert isinstance(result, str)
    assert result.lower().startswith("error")


def test_calculator_blocks_variable_reference() -> None:
    """
    Variable references (undefined names) must be blocked.

    The safe evaluator must not allow reading arbitrary names.
    """
    tool = create_calculator_tool()
    result = tool.invoke("x + 1")

    assert isinstance(result, str)
    assert result.lower().startswith("error")


def test_calculator_empty_expression() -> None:
    """An empty expression must return an error string without crashing."""
    tool = create_calculator_tool()
    result = tool.invoke("")

    assert isinstance(result, str)
    assert result.lower().startswith("error")


def test_calculator_invalid_syntax() -> None:
    """An expression with invalid Python syntax must return an error string."""
    tool = create_calculator_tool()
    result = tool.invoke("2 + * 2")

    assert isinstance(result, str)
    assert result.lower().startswith("error")


def test_calculator_float_whole_number() -> None:
    """A float result equal to a whole number must be returned without trailing '.0'."""
    tool = create_calculator_tool()
    result = tool.invoke("4.0 + 0.0")

    assert result == "4"


# ---------------------------------------------------------------------------
# get_default_tools
# ---------------------------------------------------------------------------


def test_get_default_tools_returns_list() -> None:
    """get_default_tools() must return a non-empty list of tools."""
    tools = get_default_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0


def test_get_default_tools_contains_search() -> None:
    """The default tool list must include a tool named 'web_search'."""
    tools = get_default_tools()
    names = [t.name for t in tools]

    assert "web_search" in names


def test_get_default_tools_contains_calculator() -> None:
    """The default tool list must include a tool named 'calculator'."""
    tools = get_default_tools()
    names = [t.name for t in tools]

    assert "calculator" in names


def test_get_default_tools_without_memory_excludes_recall() -> None:
    """Without a memory instance the recall_history tool must not be included."""
    tools = get_default_tools(memory=None)
    names = [t.name for t in tools]

    assert "recall_history" not in names


def test_get_default_tools_with_memory_includes_recall() -> None:
    """When a ConversationMemory instance is provided, recall_history must be included."""
    from core.memory import ConversationMemory

    with ConversationMemory(":memory:") as mem:
        tools = get_default_tools(memory=mem)
        names = [t.name for t in tools]

    assert "recall_history" in names


# ---------------------------------------------------------------------------
# Calculator: Pow guard and builtins
# ---------------------------------------------------------------------------


class TestCalculatorPowGuard:
    """Tests for the calculator exponentiation guard and built-in functions."""

    def test_large_exponent_rejected(self) -> None:
        """Calculator should reject exponents larger than 1000."""
        tools = get_default_tools()
        calc = next(t for t in tools if t.name == "calculator")
        result = calc.invoke({"expression": "2 ** 1001"})
        assert "error" in result.lower() or "too large" in result.lower()

    def test_normal_exponent_allowed(self) -> None:
        """Calculator should allow reasonable exponents."""
        tools = get_default_tools()
        calc = next(t for t in tools if t.name == "calculator")
        result = calc.invoke({"expression": "2 ** 10"})
        assert "1024" in result

    def test_min_function(self) -> None:
        """Calculator should support min()."""
        tools = get_default_tools()
        calc = next(t for t in tools if t.name == "calculator")
        result = calc.invoke({"expression": "min(3, 1, 2)"})
        assert "1" in result

    def test_max_function(self) -> None:
        """Calculator should support max()."""
        tools = get_default_tools()
        calc = next(t for t in tools if t.name == "calculator")
        result = calc.invoke({"expression": "max(3, 1, 2)"})
        assert "3" in result


# ---------------------------------------------------------------------------
# recall_history tool
# ---------------------------------------------------------------------------


class TestRecallHistoryTool:
    """Tests for the recall_history tool created by create_memory_tool."""

    def test_recall_history_no_runs(self) -> None:
        """recall_history returns 'No previous run history' for an empty memory."""
        from core.memory import ConversationMemory
        from core.tools import create_memory_tool

        with ConversationMemory(":memory:") as mem:
            tool = create_memory_tool(mem)
            result = tool.invoke("any query")

        assert "No previous run history" in result

    def test_recall_history_with_runs(self) -> None:
        """recall_history returns formatted run records when runs exist."""
        from core.memory import ConversationMemory
        from core.tools import create_memory_tool

        with ConversationMemory(":memory:") as mem:
            mem.save_run(
                run_id="run-001",
                query="What is AI?",
                result={"summary": "AI is intelligence demonstrated by machines."},
                metadata={"session_id": "s1"},
            )
            mem.save_run(
                run_id="run-002",
                query="What is ML?",
                result={"executive_summary": "ML is a subset of AI."},
                metadata={"session_id": "s1"},
            )
            tool = create_memory_tool(mem)
            result = tool.invoke("previous research")

        assert "2 recent run(s)" in result
        assert "run-001" in result
        assert "run-002" in result
        assert "AI is intelligence" in result
        assert "ML is a subset" in result

    def test_recall_history_handles_memory_error(self) -> None:
        """recall_history returns an error message when list_runs raises."""
        from unittest.mock import MagicMock

        from core.tools import create_memory_tool

        mock_mem = MagicMock()
        mock_mem.list_runs.side_effect = RuntimeError("DB gone")

        tool = create_memory_tool(mock_mem)
        result = tool.invoke("anything")

        assert "Error retrieving history" in result
        assert "DB gone" in result

    def test_recall_history_result_without_summary_key(self) -> None:
        """recall_history handles runs whose result has no recognisable summary key."""
        from core.memory import ConversationMemory
        from core.tools import create_memory_tool

        with ConversationMemory(":memory:") as mem:
            mem.save_run(
                run_id="run-003",
                query="Test query",
                result={"custom_field": "custom_value"},
                metadata={},
            )
            tool = create_memory_tool(mem)
            result = tool.invoke("check")

        assert "1 recent run(s)" in result
        assert "no summary available" in result


# ---------------------------------------------------------------------------
# Search provider branches (Tavily / SerpAPI)
# ---------------------------------------------------------------------------


class TestSearchProviderBranches:
    """Tests for the Tavily and SerpAPI creation branches in create_search_tool."""

    def test_tavily_provider_success(self) -> None:
        """create_search_tool returns a Tavily tool when SEARCH_PROVIDER=tavily."""
        import os
        from unittest.mock import MagicMock, patch

        mock_tavily_cls = MagicMock()
        mock_tavily_instance = MagicMock()
        mock_tavily_cls.return_value = mock_tavily_instance

        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilySearchResults = mock_tavily_cls

        mock_community = MagicMock()
        mock_community.tools = MagicMock()
        mock_community.tools.tavily_search = mock_tavily_module

        with (
            patch.dict(
                os.environ,
                {"SEARCH_PROVIDER": "tavily", "TAVILY_API_KEY": "test-key"},
            ),
            patch.dict(
                "sys.modules",
                {
                    "langchain_community": mock_community,
                    "langchain_community.tools": mock_community.tools,
                    "langchain_community.tools.tavily_search": mock_tavily_module,
                },
            ),
        ):
            result = create_search_tool()

        assert result is mock_tavily_instance

    def test_tavily_provider_import_error(self) -> None:
        """create_search_tool raises ImportError when tavily package is missing."""
        import os
        from unittest.mock import patch

        with (
            patch.dict(os.environ, {"SEARCH_PROVIDER": "tavily"}),
            patch.dict(
                "sys.modules",
                {
                    "langchain_community": None,
                    "langchain_community.tools": None,
                    "langchain_community.tools.tavily_search": None,
                },
            ),
        ):
            import pytest

            with pytest.raises(ImportError, match="Tavily"):
                create_search_tool()

    def test_serpapi_provider_success(self) -> None:
        """create_search_tool returns a SerpAPI tool when SEARCH_PROVIDER=serpapi."""
        import os
        from unittest.mock import MagicMock, patch

        mock_wrapper_cls = MagicMock()
        mock_wrapper_instance = MagicMock()
        mock_wrapper_instance.run = MagicMock()
        mock_wrapper_cls.return_value = mock_wrapper_instance

        mock_utilities_module = MagicMock()
        mock_utilities_module.SerpAPIWrapper = mock_wrapper_cls

        mock_community = MagicMock()
        mock_community.utilities = mock_utilities_module

        with (
            patch.dict(
                os.environ,
                {"SEARCH_PROVIDER": "serpapi", "SERPAPI_API_KEY": "test-key"},
            ),
            patch.dict(
                "sys.modules",
                {
                    "langchain_community": mock_community,
                    "langchain_community.utilities": mock_utilities_module,
                },
            ),
        ):
            result = create_search_tool()

        assert result.name == "web_search"

    def test_serpapi_provider_import_error(self) -> None:
        """create_search_tool raises ImportError when serpapi package is missing."""
        import os
        from unittest.mock import patch

        with (
            patch.dict(os.environ, {"SEARCH_PROVIDER": "serpapi"}),
            patch.dict(
                "sys.modules",
                {
                    "langchain_community": None,
                    "langchain_community.utilities": None,
                },
            ),
        ):
            import pytest

            with pytest.raises(ImportError, match="SerpAPI"):
                create_search_tool()

    def test_unknown_provider_falls_back_to_mock(self) -> None:
        """An unknown SEARCH_PROVIDER falls back to the mock tool with a warning."""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"SEARCH_PROVIDER": "unknown_provider"}):
            tool = create_search_tool()

        assert tool.name == "web_search"
        result = tool.invoke("test query")
        assert isinstance(result, str)
        assert len(result) > 0
