# Compliance Considerations — Financial Memo

**This pack produces assistive strategy memos only. It is not financial or investment advice.**

## Regulatory context (non-exhaustive)

| Framework | Relevance |
|-----------|-----------|
| **MiFID II (EU)** | Content that recommends financial instruments may qualify as investment advice or an investment recommendation, triggering suitability, disclosure, and record-keeping duties for the operator. |
| **SEC Investment Advisers Act / FINRA Rule 2210 (US)** | Advice or communications about securities are regulated; AI-drafted memos used with clients may fall in scope. |
| **Market abuse / research rules** | Memos that circulate views on listed instruments may be treated as investment research, with conflict-of-interest and labelling requirements. |

## What this template provides

- Pack disabled by default: requires explicit `REGULATED_PACKS_ENABLED=true` opt-in (`domain_packs/common/compliance.py`).
- `human_review_required: true` and the mandatory not-financial-advice `disclaimer` injected server-side on every response — the LLM cannot omit or weaken them (`apply_compliance_output`).
- `PackPolicy` with `human_review_required=True` and a 2 000-character input cap (`control_plane/__init__.py`).
- Prompt-injection delimiters around all untrusted input fields (`domain_packs/common/prompt_safety.py`).
- Strict Pydantic output schema (`extra="forbid"`, strict validation for regulated packs).
- Output integrity scan with structured audit logging of suspicious findings (`domain_packs/common/output_guard.py`); an optional second-LLM cross-check can be enabled via `PackPolicy.extensions["output_cross_check"]`.

## What operators must still implement

- Review by a qualified finance professional of every figure, assumption, and recommendation before any decision or client use.
- A determination of whether outputs constitute regulated investment advice/research in your jurisdiction, and the corresponding licensing, suitability, and disclosure controls.
- Audit logging, retention policies, and access controls for memo inputs and outputs (which may contain material non-public information).
- Legal/compliance sign-off before production use.

**For research and demo only unless operated under appropriate professional supervision and a full compliance programme.**
