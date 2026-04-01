"""
core/tools.py — Reusable LangChain tools for the LangGraph agent stack.

Every tool in this module is framework-agnostic: they accept a plain string
input and return a plain string output, making them compatible with any
LangChain/LangGraph agent without additional wiring.

Tool catalogue
--------------
``create_search_tool``
    Web/document search.  Ships with a deterministic mock by default.
    Override by setting ``SEARCH_PROVIDER=tavily`` or ``SEARCH_PROVIDER=serpapi``
    in the environment (requires the matching SDK).

``create_calculator_tool``
    Safe arithmetic expression evaluator using Python's ``ast`` module.
    No ``eval()`` — only numeric literals and the four basic operators plus
    ``**``, ``%``, ``//``, ``abs()``, ``round()``, ``int()``, ``float()``.

``create_memory_tool``
    Reads recent run history from a ``ConversationMemory`` instance.
    Accepts a natural-language query and returns the N most-recent runs as
    formatted text.

``get_default_tools``
    Convenience helper that assembles the default tool list in one call.

Usage example::

    from core.memory import ConversationMemory
    from core.tools import get_default_tools

    memory = ConversationMemory("./data/memory.db")
    tools = get_default_tools(memory=memory)
    # Pass tools to a BaseAgent subclass or directly to an LLM with tool use
"""

from __future__ import annotations

import ast
import logging
import operator
import os
from typing import Any
from urllib.parse import quote_plus

from langchain_core.tools import BaseTool, tool

from core.memory import ConversationMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search tool
# ---------------------------------------------------------------------------

# Environment variable controlling which search provider to use.
_SEARCH_PROVIDER_ENV = "SEARCH_PROVIDER"
_TAVILY_API_KEY_ENV = "TAVILY_API_KEY"
_SERPAPI_API_KEY_ENV = "SERPAPI_API_KEY"


def create_search_tool() -> BaseTool:
    """
    Create a web-search LangChain tool.

    Provider resolution order (first matching wins):

    1. ``SEARCH_PROVIDER=tavily`` + ``TAVILY_API_KEY`` set → ``TavilySearchResults``
    2. ``SEARCH_PROVIDER=serpapi`` + ``SERPAPI_API_KEY`` set → ``SerpAPIWrapper``
    3. Fallback → deterministic mock that returns templated strings

    The mock is intentionally stable so unit tests do not require real API keys.
    Replace the mock by setting the appropriate environment variables in
    production.

    Returns:
        A ``BaseTool`` instance whose ``.invoke(query)`` accepts a string
        query and returns a string with one or more result snippets.

    Raises:
        ImportError: Only when the requested provider package is missing —
            the exception message contains the exact ``pip install`` command.
    """
    provider = os.environ.get(_SEARCH_PROVIDER_ENV, "mock").lower().strip()

    if provider == "tavily":
        return _create_tavily_tool()

    if provider == "serpapi":
        return _create_serpapi_tool()

    if provider != "mock":
        logger.warning("Unknown SEARCH_PROVIDER %r — using mock search tool.", provider)

    return _create_mock_search_tool()


def _create_tavily_tool() -> BaseTool:
    """
    Build a ``TavilySearchResults`` tool.

    Requires:
        * ``pip install tavily-python langchain-community``
        * ``TAVILY_API_KEY`` environment variable set.

    Returns:
        A configured ``TavilySearchResults`` LangChain tool.

    Raises:
        ImportError: When ``langchain_community`` is not installed.
    """
    try:
        from langchain_community.tools.tavily_search import (  # type: ignore[import]
            TavilySearchResults,
        )

        api_key = os.environ.get(_TAVILY_API_KEY_ENV, "")
        tavily_tool = TavilySearchResults(
            max_results=5,
            tavily_api_key=api_key or None,
        )
        logger.info("Search tool: TavilySearchResults initialised")
        return tavily_tool

    except ImportError as exc:
        raise ImportError(
            "Tavily search requires langchain-community: "
            "pip install tavily-python langchain-community"
        ) from exc


def _create_serpapi_tool() -> BaseTool:
    """
    Build a SerpAPI search tool wrapped as a LangChain ``Tool``.

    Requires:
        * ``pip install google-search-results langchain-community``
        * ``SERPAPI_API_KEY`` environment variable set.

    Returns:
        A ``Tool`` wrapping ``SerpAPIWrapper``.

    Raises:
        ImportError: When ``langchain_community`` is not installed.
    """
    try:
        from langchain_community.utilities import SerpAPIWrapper  # type: ignore[import]
        from langchain_core.tools import Tool  # type: ignore[import]

        api_key = os.environ.get(_SERPAPI_API_KEY_ENV, "")
        search = SerpAPIWrapper(serpapi_api_key=api_key or None)
        serp_tool = Tool(
            name="web_search",
            description=(
                "Search the web for current information. "
                "Input: a natural language search query. "
                "Output: a string with the top search results."
            ),
            func=search.run,
        )
        logger.info("Search tool: SerpAPIWrapper initialised")
        return serp_tool

    except ImportError as exc:
        raise ImportError(
            "SerpAPI search requires langchain-community: "
            "pip install google-search-results langchain-community"
        ) from exc


def _create_mock_search_tool() -> BaseTool:
    """
    Build a deterministic mock search tool for development and testing.

    The mock returns a fixed two-result template with the query embedded,
    making responses inspectable and stable across test runs.

    Returns:
        A ``BaseTool`` whose ``.invoke(query)`` returns two mock snippets.
    """

    @tool
    def web_search(query: str) -> str:
        """
        Search the web for information on a given topic.

        Args:
            query: The natural language search query to execute.

        Returns:
            A string containing one or more result snippets with source URLs.
        """
        logger.debug("Mock search invoked", extra={"query": query[:120]})
        safe_query = quote_plus(query)
        safe_slug = quote_plus(query.lower())
        return (
            f"[MOCK RESULT 1] Comprehensive overview of '{query}': "
            "This representative snippet describes key aspects of the topic, "
            "covering background, current state, and notable developments.\n"
            f"Source: https://example.com/search?q={safe_query}\n\n"
            f"[MOCK RESULT 2] Recent findings on '{query}': "
            "Secondary perspective offering additional context and alternative "
            "viewpoints for a more balanced understanding.\n"
            f"Source: https://news.example.com/{safe_slug}"
        )

    logger.info("Search tool: mock implementation active")
    return web_search  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Calculator tool
# ---------------------------------------------------------------------------

# Allowed AST node types for the safe evaluator
_SAFE_NODES = (
    ast.Module,
    ast.Expr,
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Call,
    ast.Name,
    ast.Load,
)

# Allowed built-in function names in expressions
_SAFE_NAMES: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "int": int,
    "float": float,
    "min": min,
    "max": max,
}

_SAFE_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> Any:
    """
    Recursively evaluate a safe arithmetic AST node.

    Only ``_SAFE_NODES`` are permitted.  Any attempt to use attribute access,
    imports, subscripts, or non-whitelisted names raises ``ValueError``.

    Args:
        node: An ``ast.AST`` node to evaluate.

    Returns:
        The numeric result of the expression.

    Raises:
        ValueError: When the expression contains disallowed constructs.
        ZeroDivisionError: When the expression divides by zero.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported literal type: {type(node.value).__name__}")
        return node.value

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _SAFE_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval(node.operand)
        return _SAFE_OPERATORS[op_type](operand)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed.")
        name = node.func.id
        if name not in _SAFE_NAMES:
            raise ValueError(
                f"Function '{name}' is not allowed. "
                f"Allowed functions: {sorted(_SAFE_NAMES.keys())}"
            )
        args = [_safe_eval(arg) for arg in node.args]
        return _SAFE_NAMES[name](*args)

    if isinstance(node, ast.Name):
        raise ValueError(
            f"Variable references are not allowed: '{node.id}'. "
            "Use only numeric literals and the allowed built-in functions."
        )

    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def create_calculator_tool() -> BaseTool:
    """
    Create a safe arithmetic calculator LangChain tool.

    The calculator parses the expression with Python's ``ast`` module and
    evaluates only whitelisted node types.  No ``eval()`` or ``exec()`` is
    used, so arbitrary code execution is not possible.

    Supported constructs:
        * Integer and float literals
        * Operators: ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``
        * Functions: ``abs()``, ``round()``, ``int()``, ``float()``,
          ``min()``, ``max()``
        * Nested parentheses

    Returns:
        A ``BaseTool`` that accepts an arithmetic expression string and
        returns the computed result as a string.
    """

    @tool
    def calculator(expression: str) -> str:
        """
        Evaluate a safe arithmetic expression and return the result.

        Supported operators: +, -, *, /, //, %, **
        Supported functions: abs(), round(), int(), float(), min(), max()

        Args:
            expression: A numeric arithmetic expression string,
                e.g. ``"(3 + 4) * 2"`` or ``"round(3.14159, 2)"``.

        Returns:
            The numeric result as a string, or an error message prefixed
            with ``"Error:"`` when the expression cannot be evaluated.
        """
        expr = expression.strip()
        if not expr:
            return "Error: empty expression."

        try:
            tree = ast.parse(expr, mode="eval")
            result = _safe_eval(tree)
            logger.debug(
                "Calculator evaluated",
                extra={"expression": expr, "result": result},
            )
            # Format: suppress trailing ".0" for whole-number float results
            if isinstance(result, float) and result == int(result):
                return str(int(result))
            return str(result)

        except ZeroDivisionError:
            return "Error: division by zero."
        except (ValueError, TypeError) as exc:
            return f"Error: {exc}"
        except SyntaxError:
            return f"Error: invalid expression syntax — {expr!r}"
        except Exception as exc:
            logger.warning(
                "Calculator unexpected error",
                extra={"expression": expr, "error": str(exc)},
            )
            return f"Error: {exc}"

    return calculator  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Memory tool
# ---------------------------------------------------------------------------


def create_memory_tool(memory: ConversationMemory) -> BaseTool:
    """
    Create a LangChain tool that reads recent run history from ``memory``.

    The tool is useful when an agent needs to check whether a similar query
    has already been answered, or when building a multi-turn conversational
    interface that should reference previous sessions.

    Args:
        memory: An initialised ``ConversationMemory`` instance.  The tool
            holds a reference to this instance — closing the memory object
            externally while an agent is running will cause errors.

    Returns:
        A ``BaseTool`` that accepts a natural-language query (used only as a
        label / hint) and returns the N most-recent run records formatted as
        readable text.
    """

    @tool
    def recall_history(query: str) -> str:
        """
        Recall the most recent agent run records from conversation history.

        Use this tool to check whether the user has asked a similar question
        before, or to reference prior research and analysis results.

        Args:
            query: A natural language description of what you are looking for
                (e.g. "previous research on vector databases").  This is used
                as a label — the tool always returns the 5 most recent runs.

        Returns:
            A formatted string listing recent run records, or a message
            indicating that no history is available.
        """
        logger.debug(
            "recall_history invoked",
            extra={"query_hint": query[:80]},
        )

        try:
            runs = memory.list_runs(limit=5)
        except Exception as exc:
            logger.warning(
                "recall_history: list_runs failed",
                extra={"error": str(exc)},
            )
            return f"Error retrieving history: {exc}"

        if not runs:
            return "No previous run history found."

        lines: list[str] = [f"Found {len(runs)} recent run(s) in history:\n"]
        for i, run in enumerate(runs, start=1):
            result_preview = ""
            result = run.get("result", {})
            if isinstance(result, dict):
                # Best-effort extraction of a human-readable summary
                for key in ("summary", "executive_summary", "output", "text"):
                    val = result.get(key, "")
                    if val:
                        result_preview = str(val)[:200]
                        break
            lines.append(
                f"[{i}] run_id={run['run_id']}\n"
                f"    created_at : {run['created_at']}\n"
                f"    query      : {run['query'][:120]}\n"
                f"    result     : {result_preview or '(no summary available)'}\n"
            )

        return "\n".join(lines)

    return recall_history  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Default tool list
# ---------------------------------------------------------------------------


def get_default_tools(
    memory: ConversationMemory | None = None,
) -> list[BaseTool]:
    """
    Return the standard set of tools used by agents in this stack.

    The list always includes ``web_search`` and ``calculator``.  The
    ``recall_history`` memory tool is added only when ``memory`` is provided,
    since it requires an initialised ``ConversationMemory`` instance.

    Args:
        memory: Optional ``ConversationMemory`` instance.  When supplied,
            a ``recall_history`` tool is appended to the list.

    Returns:
        A list of ``BaseTool`` instances ready to be bound to an LLM or
        passed to a LangGraph agent node.
    """
    tools: list[BaseTool] = [
        create_search_tool(),
        create_calculator_tool(),
    ]

    if memory is not None:
        tools.append(create_memory_tool(memory))
        logger.debug(
            "get_default_tools: memory tool included",
            extra={"tool_count": len(tools)},
        )
    else:
        logger.debug(
            "get_default_tools: no memory provided — recall_history excluded",
            extra={"tool_count": len(tools)},
        )

    return tools
