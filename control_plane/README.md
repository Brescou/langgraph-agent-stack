# Control plane

Pack-level policies are registered explicitly in `control_plane/__init__.py` (Approach B, same spirit as `PackRegistry`).

## What is enforced today

| Mechanism | Where |
|-----------|--------|
| `max_query_chars` | `validate_query_for_pack()` — used by API before pack and legacy runs |
| `budget_usd_ceiling` | `effective_budget_usd()` — passed as `budget_usd` when constructing packs (overridden by `PACK_DEFAULT_BUDGET_USD`) |
| `stream_timeout_seconds` | `effective_stream_timeout_seconds()` — caps SSE timeouts per pack route and legacy `/run/stream` |
| `human_review_required` | `domain_packs.common.compliance.apply_compliance_output()` — injects mandatory `disclaimer` + `human_review_required` on regulated pack outputs |
| `guardrails` (`GuardrailPolicy`) | `guardrail_rule_sets()` — names the rule sets screened on input and output when `GUARDRAILS_ENABLED=true`; `validate_guardrail_policies()` fails startup on an unknown set. Empty on every built-in policy. See `docs/security.md` → Guardrails |

Regulated packs (HR, contract review, financial memo) set `human_review_required=True`. See each pack's `COMPLIANCE.md`.

## Registry

```python
from control_plane import (
    ExecutionConstraints,
    GuardrailPolicy,
    PackPolicy,
    PolicyRegistry,
)

PolicyRegistry.register(
    PackPolicy(
        pack_id="my_pack",
        constraints=ExecutionConstraints(max_query_chars=1500, budget_usd_ceiling=0.50),
        guardrails=GuardrailPolicy(
            input_rule_sets=("prompt_injection",),
            output_rule_sets=("pii_basic",),
        ),
    )
)
```

## What is still not here

- Dynamic policy DSL / OPA
- Multi-tenant quotas
- Per-tenant or named API keys (built-in auth is one shared `API_KEY`; see `docs/security.md`)
- Automatic pack registration from policies (policies reference existing `pack_id` values only)
