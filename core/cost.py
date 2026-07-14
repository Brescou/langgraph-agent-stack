"""
core/cost.py — LLM cost tracking and budget enforcement.

This module provides:

1. **Per-model pricing table** with prefix/alias resolution and optional JSON
   override via ``LLM_COST_TABLE_PATH`` / ``Settings.llm_cost_table_path``.

2. **compute_call_cost()** — pricing with prompt-cache and batch API discounts.

3. **BudgetExceededError** / **UnknownModelPricingError** — budget and pricing
   guardrails (unknown models fail fast when ``ENVIRONMENT=production``).

4. **CostTracker** — LangChain callback: pre-call budget estimate on
   ``on_chat_model_start``, actual cost on ``on_llm_end``, Prometheus counter.

5. **load_cost_table()** — merges built-in pricing with an external JSON file.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

_log = logging.getLogger(__name__)

_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")
_BEDROCK_VERSION_RE = re.compile(r"-v\d+:\d+$")
_DEFAULT_MAX_OUTPUT_TOKENS = 4096
_CHARS_PER_TOKEN_ESTIMATE = 4


# ---------------------------------------------------------------------------
# Pricing model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPricing:
    """USD rates per 1k tokens for a model family."""

    input_per_1k: float
    output_per_1k: float
    cache_read_per_1k: float | None = None
    cache_write_per_1k: float | None = None
    batch_multiplier: float = 0.5

    def cache_read_rate(self) -> float:
        """Rate for cache hits (default: 90% discount vs uncached input)."""
        if self.cache_read_per_1k is not None:
            return self.cache_read_per_1k
        return self.input_per_1k * 0.1

    def cache_write_rate(self) -> float:
        """Rate for cache creation tokens (Anthropic-style 125% of input)."""
        if self.cache_write_per_1k is not None:
            return self.cache_write_per_1k
        return self.input_per_1k * 1.25


def _pricing(
    input_per_1k: float,
    output_per_1k: float,
    *,
    cache_read_per_1k: float | None = None,
    cache_write_per_1k: float | None = None,
    batch_multiplier: float = 0.5,
) -> ModelPricing:
    return ModelPricing(
        input_per_1k=input_per_1k,
        output_per_1k=output_per_1k,
        cache_read_per_1k=cache_read_per_1k,
        cache_write_per_1k=cache_write_per_1k,
        batch_multiplier=batch_multiplier,
    )


# Built-in pricing (USD per 1k tokens). Override via LLM_COST_TABLE_PATH.
COST_PER_1K: dict[str, ModelPricing] = {
    # Anthropic — Claude 3.x
    "claude-3-5-sonnet-20241022": _pricing(0.003, 0.015),
    "claude-3-5-haiku-20241022": _pricing(0.0008, 0.004),
    "claude-3-opus-20240229": _pricing(0.015, 0.075),
    "claude-3-haiku-20240307": _pricing(0.00025, 0.00125),
    # Anthropic — Claude 4.x / 4.5
    "claude-sonnet-4-20250514": _pricing(0.003, 0.015),
    "claude-opus-4-20250514": _pricing(0.015, 0.075),
    "claude-opus-4-1": _pricing(0.015, 0.075),
    "claude-opus-4-7": _pricing(0.015, 0.075),
    "claude-haiku-4-5": _pricing(0.001, 0.005),
    # Anthropic — Claude 5 generation (current, July 2026)
    # Sticker price; an introductory $2/$10 per MTok rate applies through
    # 2026-08-31, so actual spend may be ~33% lower until then.
    "claude-sonnet-5": _pricing(0.003, 0.015),
    "claude-opus-4-8": _pricing(0.005, 0.025),
    # OpenAI — GPT-4o family
    "gpt-4o": _pricing(0.005, 0.015),
    "gpt-4o-mini": _pricing(0.00015, 0.0006),
    "gpt-4o-2024-08-06": _pricing(0.005, 0.015),
    # OpenAI — GPT-4 / turbo
    "gpt-4-turbo": _pricing(0.01, 0.03),
    "gpt-4-turbo-preview": _pricing(0.01, 0.03),
    "gpt-4": _pricing(0.03, 0.06),
    # OpenAI — GPT-4.1
    "gpt-4.1": _pricing(0.002, 0.008),
    "gpt-4.1-mini": _pricing(0.0004, 0.0016),
    "gpt-4.1-nano": _pricing(0.0001, 0.0004),
    # OpenAI — reasoning
    "o1": _pricing(0.015, 0.06),
    "o1-mini": _pricing(0.003, 0.012),
    "o1-preview": _pricing(0.015, 0.06),
    "o3": _pricing(0.01, 0.04),
    "o3-mini": _pricing(0.0011, 0.0044),
    # OpenAI — GPT-5.x generation (current, July 2026)
    "gpt-5.5": _pricing(0.005, 0.03),
    # Google Gemini
    "gemini-1.5-pro": _pricing(0.00125, 0.005),
    "gemini-1.5-flash": _pricing(0.000075, 0.0003),
    "gemini-2.0-flash": _pricing(0.0001, 0.0004),
    "gemini-2.5-pro": _pricing(0.00125, 0.01),
    "gemini-2.5-flash": _pricing(0.00015, 0.0006),
    # Google Gemini 3.x generation (current, July 2026)
    "gemini-3.5-flash": _pricing(0.0015, 0.009),
    # Mistral
    "mistral-large-latest": _pricing(0.002, 0.006),
    "mistral-small-latest": _pricing(0.0002, 0.0006),
    "pixtral-large-latest": _pricing(0.002, 0.006),
    # Meta Llama (hosted inference list prices — override in prod as needed)
    "llama-3.3-70b-instruct": _pricing(0.00059, 0.00079),
    "llama-3.1-405b-instruct": _pricing(0.003, 0.003),
    "llama-3.1-70b-instruct": _pricing(0.00059, 0.00079),
    "llama-3.1-8b-instruct": _pricing(0.00005, 0.00008),
    # Mock provider — zero cost so CostTracker / metrics run without spend.
    "mock-provider": _pricing(0.0, 0.0),
}

# Longest-prefix-first aliases for dated / variant model IDs.
MODEL_ID_PREFIX_ALIASES: tuple[tuple[str, str], ...] = (
    ("claude-sonnet-4", "claude-sonnet-4-20250514"),
    ("claude-opus-4-1", "claude-opus-4-1"),
    ("claude-opus-4-8", "claude-opus-4-8"),
    ("claude-opus-4", "claude-opus-4-20250514"),
    ("claude-haiku-4-5", "claude-haiku-4-5"),
    ("claude-sonnet-5", "claude-sonnet-5"),
    ("claude-3-5-sonnet", "claude-3-5-sonnet-20241022"),
    ("claude-3-5-haiku", "claude-3-5-haiku-20241022"),
    ("claude-3-opus", "claude-3-opus-20240229"),
    ("claude-3-haiku", "claude-3-haiku-20240307"),
    ("gpt-4.1-nano", "gpt-4.1-nano"),
    ("gpt-4.1-mini", "gpt-4.1-mini"),
    ("gpt-4.1", "gpt-4.1"),
    ("gpt-4o-mini", "gpt-4o-mini"),
    ("gpt-4o", "gpt-4o"),
    ("gpt-4-turbo", "gpt-4-turbo"),
    ("o3-mini", "o3-mini"),
    ("o3", "o3"),
    ("o1-mini", "o1-mini"),
    ("o1", "o1"),
    ("gemini-2.5-pro", "gemini-2.5-pro"),
    ("gemini-2.5-flash", "gemini-2.5-flash"),
    ("gemini-2.0-flash", "gemini-2.0-flash"),
    ("gemini-1.5-pro", "gemini-1.5-pro"),
    ("gemini-1.5-flash", "gemini-1.5-flash"),
    ("anthropic.claude-3-5-sonnet", "claude-3-5-sonnet-20241022"),
    ("anthropic.claude-3-5-haiku", "claude-3-5-haiku-20241022"),
    ("anthropic.claude-3-opus", "claude-3-opus-20240229"),
    ("anthropic.claude-sonnet-4", "claude-sonnet-4-20250514"),
    ("anthropic.claude-opus-4-8", "claude-opus-4-8"),
    ("anthropic.claude-opus-4", "claude-opus-4-20250514"),
    ("anthropic.claude-haiku-4-5", "claude-haiku-4-5"),
    ("anthropic.claude-sonnet-5", "claude-sonnet-5"),
    ("meta.llama3-3-70b-instruct", "llama-3.3-70b-instruct"),
    ("meta.llama3-1-405b-instruct", "llama-3.1-405b-instruct"),
    ("meta.llama3-1-70b-instruct", "llama-3.1-70b-instruct"),
    ("meta.llama3-1-8b-instruct", "llama-3.1-8b-instruct"),
    ("mistral-large", "mistral-large-latest"),
    ("mistral-small", "mistral-small-latest"),
)

# ---------------------------------------------------------------------------
# Prometheus counter (optional — exact pattern from core/observability.py)
# ---------------------------------------------------------------------------

pack_run_cost_usd_total: Any | None
llm_cost_usd_total: Any | None
try:
    from prometheus_client import Counter

    pack_run_cost_usd_total = Counter(
        "pack_run_cost_usd_total",
        "Cumulative LLM cost in USD per model",
        ["model"],
    )
    # ``provider`` keeps cardinality lower than per-model labels: a handful of
    # vendors versus dozens of dated model IDs.
    llm_cost_usd_total = Counter(
        "llm_cost_usd_total",
        "Cumulative LLM cost in USD per provider",
        ["provider"],
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    pack_run_cost_usd_total = None
    llm_cost_usd_total = None
    _PROMETHEUS_AVAILABLE = False


# Longest-prefix-first mapping from normalised model IDs to provider labels.
_MODEL_PROVIDER_PREFIXES: tuple[tuple[str, str], ...] = (
    ("anthropic.", "anthropic"),
    ("claude", "anthropic"),
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("gemini", "google"),
    ("mistral", "mistral"),
    ("pixtral", "mistral"),
    ("meta.llama", "meta"),
    ("llama", "meta"),
    ("mock-provider", "mock"),
    ("mock", "mock"),
)


def provider_from_model_id(model_id: str) -> str:
    """Map a model ID to a low-cardinality provider label.

    Bedrock-style IDs (``us.anthropic.claude-...``) are reduced to their
    vendor fragment first.  Unrecognised IDs map to ``"unknown"`` so the
    label set stays bounded.
    """
    normalised = _normalize_model_id(model_id)
    if not normalised:
        return "unknown"
    candidates = [normalised]
    bedrock = _bedrock_model_fragment(normalised)
    if bedrock:
        candidates.append(bedrock)
    for prefix, provider in _MODEL_PROVIDER_PREFIXES:
        for candidate in candidates:
            if candidate.startswith(prefix):
                return provider
    return "unknown"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BudgetExceededError(Exception):
    """Raised by CostTracker when accumulated spend exceeds the budget ceiling."""

    def __init__(self, budget: float, actual: float) -> None:
        super().__init__(
            f"Budget ceiling ${budget:.4f} exceeded (this run: ${actual:.4f})"
        )
        self.budget = budget
        self.actual = actual


class UnknownModelPricingError(Exception):
    """Raised when pricing is missing for a model in strict (production) mode."""

    def __init__(self, model_id: str) -> None:
        super().__init__(
            f"No pricing entry for model '{model_id}'. "
            "Add it to LLM_COST_TABLE_PATH or extend COST_PER_1K in core/cost.py."
        )
        self.model_id = model_id


# ---------------------------------------------------------------------------
# load_cost_table
# ---------------------------------------------------------------------------


def _parse_pricing_entry(raw: Any) -> ModelPricing:
    if isinstance(raw, ModelPricing):
        return raw
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError(f"Invalid pricing entry: {raw!r}")
    input_rate = float(raw[0])
    output_rate = float(raw[1])
    cache_read = float(raw[2]) if len(raw) > 2 else None
    cache_write = float(raw[3]) if len(raw) > 3 else None
    batch_mult = float(raw[4]) if len(raw) > 4 else 0.5
    return _pricing(
        input_rate,
        output_rate,
        cache_read_per_1k=cache_read,
        cache_write_per_1k=cache_write,
        batch_multiplier=batch_mult,
    )


def load_cost_table(path: Path | None = None) -> dict[str, ModelPricing]:
    """Return the effective pricing table, optionally merged with an external file."""
    if path is None or not path.exists():
        return dict(COST_PER_1K)

    try:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        overrides = {k: _parse_pricing_entry(v) for k, v in raw.items()}
        merged = dict(COST_PER_1K)
        merged.update(overrides)
        return merged
    except Exception as exc:  # noqa: BLE001
        _log.warning(
            "Failed to load external cost table from %s: %s — using built-in table",
            path,
            exc,
        )
        return dict(COST_PER_1K)


_effective_table: dict[str, ModelPricing] | None = None
_effective_table_path_key: str | None = None


def _reset_effective_table() -> None:
    """Reset the cached effective cost table. For testing only."""
    global _effective_table, _effective_table_path_key
    _effective_table = None
    _effective_table_path_key = None


def _resolve_cost_table_path() -> Path | None:
    """Return the configured override path, if any."""
    override_path: Path | None = None
    try:
        from core.config import get_settings

        override_path = get_settings().llm_cost_table_path
    except Exception:  # noqa: BLE001
        pass

    if override_path is None:
        import os

        env_path = os.environ.get("LLM_COST_TABLE_PATH", "")
        override_path = Path(env_path) if env_path else None
    return override_path


def _get_effective_table() -> dict[str, ModelPricing]:
    """Return the cost table, lazily loading an override file if configured."""
    global _effective_table, _effective_table_path_key
    path_key = str(_resolve_cost_table_path() or "")
    if _effective_table is not None and _effective_table_path_key == path_key:
        return _effective_table

    _effective_table_path_key = path_key
    _effective_table = load_cost_table(_resolve_cost_table_path())
    return _effective_table


def _strict_pricing_required() -> bool:
    try:
        from core.config import get_settings

        return get_settings().environment == "production"
    except Exception:  # noqa: BLE001
        import os

        return os.environ.get("ENVIRONMENT", "development") == "production"


def _normalize_model_id(model_id: str) -> str:
    return model_id.lower().strip()


def _bedrock_model_fragment(model_id: str) -> str | None:
    """Extract ``anthropic.claude-...`` fragment from Bedrock inference IDs."""
    if "." not in model_id:
        return None
    fragment = model_id.split(".", 1)[-1]
    fragment = _BEDROCK_VERSION_RE.sub("", fragment)
    return fragment


def resolve_model_pricing(model_id: str) -> ModelPricing | None:
    """Resolve pricing for a model ID using exact, dated, prefix, and Bedrock rules."""
    normalised = _normalize_model_id(model_id)
    if not normalised:
        return None

    table = _get_effective_table()
    candidates = [normalised]
    if _DATE_SUFFIX_RE.search(normalised):
        candidates.append(_DATE_SUFFIX_RE.sub("", normalised))
    bedrock = _bedrock_model_fragment(normalised)
    if bedrock:
        candidates.append(bedrock)
        if _DATE_SUFFIX_RE.search(bedrock):
            candidates.append(_DATE_SUFFIX_RE.sub("", bedrock))

    for candidate in candidates:
        if candidate in table:
            return table[candidate]

    for prefix, canonical in MODEL_ID_PREFIX_ALIASES:
        for candidate in candidates:
            if candidate.startswith(prefix) and canonical in table:
                return table[canonical]

    return None


@dataclass(frozen=True)
class TokenUsage:
    """Token breakdown for a single LLM call."""

    model_id: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    batch: bool = False


def compute_call_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    batch: bool = False,
    strict: bool | None = None,
) -> float:
    """Calculate USD cost for one LLM call including cache and batch discounts."""
    pricing = resolve_model_pricing(model_id)
    if pricing is None:
        if (strict if strict is not None else _strict_pricing_required()) and model_id:
            raise UnknownModelPricingError(model_id)
        if model_id:
            _log.warning(
                "Unknown model ID '%s' — cost will be reported as $0.00",
                _normalize_model_id(model_id),
            )
        return 0.0

    billed_input = max(
        0,
        input_tokens - cache_read_tokens - cache_write_tokens,
    )
    cost = (
        (billed_input / 1_000.0) * pricing.input_per_1k
        + (cache_read_tokens / 1_000.0) * pricing.cache_read_rate()
        + (cache_write_tokens / 1_000.0) * pricing.cache_write_rate()
        + (output_tokens / 1_000.0) * pricing.output_per_1k
    )
    if batch:
        cost *= pricing.batch_multiplier
    return cost


def estimate_worst_case_call_cost(
    model_id: str,
    input_tokens: int,
    max_output_tokens: int,
    *,
    strict: bool | None = None,
) -> float:
    """Conservative pre-call estimate (full input rate + max output, no cache credit)."""
    pricing = resolve_model_pricing(model_id)
    if pricing is None:
        if (strict if strict is not None else _strict_pricing_required()) and model_id:
            raise UnknownModelPricingError(model_id)
        return 0.0
    return (input_tokens / 1_000.0) * pricing.input_per_1k + (
        max_output_tokens / 1_000.0
    ) * pricing.output_per_1k


def _compute_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Backward-compatible wrapper around :func:`compute_call_cost`."""
    return compute_call_cost(model_id, input_tokens, output_tokens)


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class CostTracker(BaseCallbackHandler):
    """LangChain callback handler that accumulates LLM token usage and cost.

    A single instance may be shared by several agents within one pipeline run
    so that the budget applies to the *cumulative* spend of the run.  Counter
    mutations are guarded by a ``threading.Lock`` because agents may execute
    concurrently (e.g. inside a ``ThreadPoolExecutor``).
    """

    def __init__(self, budget_usd: float | None = None) -> None:
        super().__init__()
        self.raise_error = True
        self.budget_usd: float | None = budget_usd
        self.total_cost_usd: float = 0.0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self._lock = threading.Lock()

    @staticmethod
    def _extract_tokens_from_result(response: LLMResult) -> TokenUsage:
        model_id = ""
        input_tokens = 0
        output_tokens = 0
        cache_read_tokens = 0
        cache_write_tokens = 0
        batch = False

        llm_output: dict[str, Any] = response.llm_output or {}
        model_id = llm_output.get("model_name", "") or llm_output.get("model", "") or ""
        batch = bool(
            llm_output.get("batch")
            or llm_output.get("batch_mode")
            or llm_output.get("is_batch")
        )

        usage_meta: dict[str, Any] | None = None
        for row in response.generations:
            for gen in row:
                message = getattr(gen, "message", None)
                if message is None:
                    continue
                if not model_id:
                    rm = getattr(message, "response_metadata", {}) or {}
                    model_id = rm.get("model_id", "") or rm.get("model", "") or ""
                usage_meta = getattr(message, "usage_metadata", None) or usage_meta

        if usage_meta:
            input_tokens = int(usage_meta.get("input_tokens", 0))
            output_tokens = int(usage_meta.get("output_tokens", 0))
            cache_read_tokens = int(
                usage_meta.get("cache_read_input_tokens", 0)
                or usage_meta.get("cached_input_tokens", 0)
            )
            cache_write_tokens = int(
                usage_meta.get("cache_creation_input_tokens", 0)
                or usage_meta.get("cache_write_input_tokens", 0)
            )
            input_details = usage_meta.get("input_token_details") or {}
            if isinstance(input_details, dict):
                cache_read_tokens = max(
                    cache_read_tokens,
                    int(input_details.get("cache_read", 0) or 0),
                    int(input_details.get("cached_tokens", 0) or 0),
                )
            batch = batch or bool(usage_meta.get("batch"))
            return TokenUsage(
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                batch=batch,
            )

        token_usage: dict[str, Any] = llm_output.get(
            "token_usage", llm_output.get("usage", {})
        )
        if token_usage:
            input_tokens = int(
                token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0))
            )
            output_tokens = int(
                token_usage.get(
                    "completion_tokens", token_usage.get("output_tokens", 0)
                )
            )
            prompt_details = token_usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict):
                cache_read_tokens = int(prompt_details.get("cached_tokens", 0) or 0)

        return TokenUsage(
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            batch=batch,
        )

    @staticmethod
    def _model_id_from_start(serialized: dict[str, Any], kwargs: dict[str, Any]) -> str:
        invocation = kwargs.get("invocation_params") or {}
        for source in (
            invocation.get("model"),
            invocation.get("model_name"),
            invocation.get("model_id"),
            (serialized.get("kwargs") or {}).get("model"),
            (serialized.get("kwargs") or {}).get("model_name"),
        ):
            if isinstance(source, str) and source.strip():
                return source.strip()
        return ""

    @staticmethod
    def _max_output_tokens_from_start(kwargs: dict[str, Any]) -> int:
        invocation = kwargs.get("invocation_params") or {}
        for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            value = invocation.get(key)
            if isinstance(value, int) and value > 0:
                return value
        return _DEFAULT_MAX_OUTPUT_TOKENS

    @staticmethod
    def _estimate_input_tokens(
        serialized: dict[str, Any],
        prompts: list[str] | None = None,
        messages: list[Any] | None = None,
    ) -> int:
        total_chars = 0
        if prompts:
            total_chars += sum(len(p) for p in prompts)
        if messages:
            for message in messages:
                content = getattr(message, "content", "")
                if isinstance(content, str):
                    total_chars += len(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            total_chars += len(str(block.get("text", "")))
                        else:
                            total_chars += len(str(block))
        if total_chars == 0:
            kwargs_messages = (serialized.get("kwargs") or {}).get("messages")
            if isinstance(kwargs_messages, list):
                for item in kwargs_messages:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        total_chars += len(str(item[1]))
        return max(1, total_chars // _CHARS_PER_TOKEN_ESTIMATE)

    def _enforce_pre_call_budget(
        self,
        model_id: str,
        input_tokens: int,
        max_output_tokens: int,
    ) -> None:
        if self.budget_usd is None:
            return
        estimate = estimate_worst_case_call_cost(
            model_id, input_tokens, max_output_tokens
        )
        with self._lock:
            projected = self.total_cost_usd + estimate
        if projected > self.budget_usd:
            raise BudgetExceededError(budget=self.budget_usd, actual=projected)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        **kwargs: Any,
    ) -> None:
        """Reject calls that cannot fit in the remaining budget (worst-case estimate)."""
        if self.budget_usd is None:
            return
        model_id = self._model_id_from_start(serialized, kwargs)
        if not model_id:
            return
        flat_messages = messages[0] if messages else []
        input_est = self._estimate_input_tokens(
            serialized, messages=list(flat_messages)
        )
        max_out = self._max_output_tokens_from_start(kwargs)
        self._enforce_pre_call_budget(model_id, input_est, max_out)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Legacy LLM path — same pre-call budget guard."""
        if self.budget_usd is None:
            return
        model_id = self._model_id_from_start(serialized, kwargs)
        if not model_id:
            return
        input_est = self._estimate_input_tokens(serialized, prompts=prompts)
        max_out = self._max_output_tokens_from_start(kwargs)
        self._enforce_pre_call_budget(model_id, input_est, max_out)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Accumulate actual token usage and cost; enforce budget after the call."""
        usage = self._extract_tokens_from_result(response)
        call_cost = compute_call_cost(
            usage.model_id,
            usage.input_tokens,
            usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            batch=usage.batch,
        )

        with self._lock:
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            self.total_cost_usd += call_cost
            total_cost_usd = self.total_cost_usd

        if _PROMETHEUS_AVAILABLE and pack_run_cost_usd_total is not None:
            pack_run_cost_usd_total.labels(model=usage.model_id or "unknown").inc(
                call_cost
            )
        if _PROMETHEUS_AVAILABLE and llm_cost_usd_total is not None:
            llm_cost_usd_total.labels(
                provider=provider_from_model_id(usage.model_id)
            ).inc(call_cost)

        _log.debug(
            "LLM call cost",
            extra={
                "model": usage.model_id,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "batch": usage.batch,
                "call_cost_usd": call_cost,
                "total_cost_usd": total_cost_usd,
            },
        )

        if self.budget_usd is not None and total_cost_usd > self.budget_usd:
            raise BudgetExceededError(
                budget=self.budget_usd,
                actual=total_cost_usd,
            )
