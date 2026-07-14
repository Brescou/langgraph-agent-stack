# Design: normalize `cost_usd` in mock mode (#88)

**Goal:** Every mock-mode route returns `cost_usd: 0.0` (not `null`), and mock runs exercise `CostTracker` via synthetic usage.

**Approach:**
1. `MockProviderChatModel` emits deterministic `usage_metadata` + model id `mock-provider`.
2. Built-in cost table prices `mock-provider` at $0/1k tokens.
3. `StructuredLLMPack` wires `CostTracker` + `cost_usd` (like research packs).
4. `api/router_factory._serialize_pack_result` injects `cost_usd` for typed pack dumps.

**Non-goals:** Change real-provider pricing; coerce `or 0.0` in endpoints.
