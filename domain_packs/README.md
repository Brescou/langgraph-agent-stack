# Domain packs

Built-in pipelines registered via `platform/builtin_packs.py` (Approach B — explicit).

## Core packs

| `pack_id` | Class | Use case |
|-----------|--------|----------|
| `research_analysis` | `ResearchAnalysisPack` | Default — Research → Analysis |
| `research_only` | `ResearchOnlyPack` | Research phase only |
| `analysis_only` | `AnalysisOnlyPack` | Analysis on pre-supplied research |
| `summariser` | `SummariserPack` | Bullet-point summary |

## Vertical packs (sellable workflows)

| `pack_id` | Route | Description |
|-----------|-------|-------------|
| `meeting_prep` | `POST /packs/meeting_prep/run` | Sales meeting brief |
| `rfp_assistant` | `POST /packs/rfp_assistant/run` | RFP analysis + response plan |
| `support_triage` | `POST /packs/support_triage/run` | Ticket triage + draft reply |
| `executive_brief` | `POST /packs/executive_brief/run` | C-level brief from long text |
| `contract_reviewer` | `POST /packs/contract_reviewer/run` | Contract risk review |
| `financial_memo` | `POST /packs/financial_memo/run` | SCQA strategy memo |

## HR packs (`domain_packs/rh/`)

| `pack_id` | Description |
|-----------|-------------|
| `talent_screening` | CV vs job description — fit score, questions, red flags |
| `job_description_writer` | Inclusive JD + rubric + bias notes |
| `hr_policy_qa` | Employee policy Q&A with citations |

## Examples

```bash
curl -X POST http://localhost:8000/packs/meeting_prep/run \
  -H 'Content-Type: application/json' \
  -d '{"company": "Acme", "person": "Jane", "meeting_goal": "discovery"}'

curl -X POST http://localhost:8000/packs/talent_screening/run \
  -H 'Content-Type: application/json' \
  -d '{"job_description": "...", "resume_text": "..."}'
```

Note: HR packs use `pack_id` without an `rh/` prefix in URLs — e.g. `/packs/talent_screening/run`.

## RAG / connectors

Packs whose constructor accepts `connector=` receive the shared connector when
`CONNECTOR_ENABLED=true` (see `api/main.py` `_pack_runtime_kwargs`). Useful for
`rfp_assistant`, `contract_reviewer`, `hr_policy_qa`, `meeting_prep`, `support_triage`.

## Authoring a new pack

1. Subclass `StructuredLLMPack` in `domain_packs/common/structured_llm.py` for single-LLM verticals, or `BaseDomainPack` for multi-agent graphs.
2. Add `schemas.py` with Pydantic input/output models.
3. Register in `platform/builtin_packs.py`.
4. Add a policy row in `control_plane/__init__.py`.
5. Tests in `tests/test_vertical_packs.py` (or dedicated file).
