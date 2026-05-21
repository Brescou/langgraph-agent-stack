# Domain packs

Built-in pipelines registered in `platform/__init__.py` (Approach B — explicit, no auto-discovery).

## Catalog

| `pack_id` | Class | Agents | Typical use |
|-----------|--------|--------|-------------|
| `research_analysis` | `ResearchAnalysisPack` | Research → Analysis | Default full pipeline (`DEFAULT_PACK_ID`) |
| `research_only` | `ResearchOnlyPack` | Research only | Collect findings without analysis |
| `analysis_only` | `AnalysisOnlyPack` | Analysis only | Analyse pre-supplied research context |
| `summariser` | `SummariserPack` | Single LLM step | Bullet-point summary of arbitrary text |

## HTTP

```bash
curl http://localhost:8000/packs
curl -X POST http://localhost:8000/packs/summariser/run \
  -H 'Content-Type: application/json' \
  -d '{"text": "Long article...", "bullet_count": 5}'
curl -X POST http://localhost:8000/packs/analysis_only/run \
  -H 'Content-Type: application/json' \
  -d '{"query": "Q?", "summary": "Prior research...", "findings": ["a","b"]}'
```

Legacy `POST /run` still targets `DEFAULT_PACK_ID` (`research_analysis`).

## Add a new pack

1. Create `domain_packs/my_pack/pack.py` subclassing `BaseDomainPack`.
2. Optional `schemas.py` for typed I/O (`input_schema` / `output_schema`).
3. Implement `run`, `arun`, `stream_events`. For non-`query` bodies, add `run_from_input` and `stream_events_from_input`.
4. Register in `platform/__init__.py`: `PackRegistry.register(MyPack)`.
5. Optional policy in `control_plane/__init__.py`.
6. Tests under `tests/` (contract + API smoke with mocks).

See `examples/custom_pack/` for a tutorial that mirrors `domain_packs/summariser/`.
