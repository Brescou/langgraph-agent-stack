# Custom Pack Example

The **SummariserPack** ships as a first-class domain pack at `domain_packs/summariser/`.

This folder remains a minimal tutorial for authoring packs outside the built-in tree.

## Register and use (tutorial copy)

```python
from platform.registry import PackRegistry
from domain_packs.summariser.pack import SummariserPack

PackRegistry.register(SummariserPack)
Pack = PackRegistry.get("summariser")
pack = Pack(run_id="my-run", llm=llm, checkpointer=checkpointer)
result = pack.run_from_input(SummaryInput(text="...", bullet_count=3))
```

## Via API (built-in registration)

```
POST /packs/summariser/run
{"text": "Your long text here...", "bullet_count": 3}
```

See `domain_packs/README.md` for the full pack catalog.
