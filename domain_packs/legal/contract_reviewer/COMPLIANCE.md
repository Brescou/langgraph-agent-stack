# Compliance Considerations — Contract Reviewer

**This pack produces assistive contract analysis only. It is not legal advice and does not replace an attorney.**

## Regulatory context (non-exhaustive)

| Framework | Relevance |
|-----------|-----------|
| **Unauthorized practice of law (UPL)** | US state UPL statutes and ABA Model Rule 5.5 restrict providing legal advice without a licence; equivalent rules exist in most jurisdictions. Operators must position output as assistive drafting, not legal counsel. |
| **Professional responsibility** | Lawyers supervising AI output remain bound by competence and supervision duties (e.g. ABA Model Rules 1.1, 5.3) and by client confidentiality/privilege obligations. |
| **GDPR / data protection** | Contracts frequently contain personal data and trade secrets; processing and retention need a lawful basis and confidentiality controls. |

## What this template provides

- Pack disabled by default: requires explicit `REGULATED_PACKS_ENABLED=true` opt-in (`domain_packs/common/compliance.py`).
- `human_review_required: true` and the mandatory not-legal-advice `disclaimer` injected server-side on every response — the LLM cannot omit or weaken them (`apply_compliance_output`).
- `PackPolicy` with `human_review_required=True` and a 500-character query cap (`control_plane/__init__.py`).
- Prompt-injection delimiters around the contract text and all untrusted input fields (`domain_packs/common/prompt_safety.py`).
- Strict Pydantic output schema (`extra="forbid"`, strict validation for regulated packs).
- Output integrity guard in **fail-closed** mode (`output_integrity_fail_closed=True`): runs whose output matches injection/manipulation patterns are rejected and audit-logged (`domain_packs/common/output_guard.py`); an optional second-LLM cross-check can be enabled via `PackPolicy.extensions["output_cross_check"]`.

## What operators must still implement

- Review by a qualified attorney of every flagged clause and recommendation before negotiation or execution.
- A UPL assessment for your jurisdiction if output is exposed to non-lawyers or external clients.
- Confidentiality, privilege, access-control, and retention measures for contract documents.
- Audit logging of who reviewed and approved each AI-assisted analysis.

**For research and demo only unless operated under appropriate legal supervision and a full compliance programme.**
