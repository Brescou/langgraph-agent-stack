# Multi-Agent Patterns

Four LangGraph patterns demonstrating different coordination strategies.

## Patterns Comparison

| Pattern | File | Use case | When to use |
|---------|------|----------|-------------|
| Sequential | `sequential/` | Research → Analysis pipeline | Clear step dependencies where each step builds on the previous output |
| Parallel | `parallel/` | Multi-angle simultaneous analysis | Independent agents with no ordering constraints; maximise throughput |
| Supervisor | `supervisor/` | Dynamic routing to specialists | Multi-domain assistant where the right specialist depends on the query |
| Human-in-loop | `human_in_loop/` | Approval before critical actions | Irreversible operations (database writes, deployments, emails) |

## Running the Examples

Prerequisites: copy `.env.example` to `.env` and set your `LLM_PROVIDER` and the matching API key.

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY (or the key for your chosen provider)
```

```bash
# Sequential
uv run python examples/sequential/graph.py

# Parallel
uv run python examples/parallel/graph.py

# Supervisor
uv run python examples/supervisor/graph.py

# Human-in-the-loop
uv run python examples/human_in_loop/graph.py
```

## Architecture Notes

All four patterns are built on the same LangGraph primitives; they differ only in how they wire nodes together.

### StateGraph

Every pattern defines a `TypedDict` state schema and constructs a `StateGraph` from it. Nodes read from and write partial updates back to that shared state. LangGraph merges the partial updates using the reducer annotations on each field (e.g. `Annotated[list, operator.add]` for append-only lists).

### Linear edges (`add_edge`)

The sequential pattern uses only `add_edge` calls to produce a straight `research_node → analyze_node → END` chain. Each node runs to completion before the next starts.

### `Send` API (dynamic fan-out)

The parallel pattern's `fan_out_node` returns a `list[Send]` instead of a state update. Each `Send("analyst_node", branch_state)` schedules an independent concurrent execution of `analyst_node`. LangGraph runs all three branches in parallel and merges their `analyses` list contributions via the `operator.add` reducer before `consolidate_node` synthesises a final report.

### Conditional edges (`add_conditional_edges`)

The supervisor pattern attaches a routing function to `add_conditional_edges`. After every supervisor call the routing function reads `next_agent` from state and returns the edge key (`"research"`, `"code"`, `"data"`, or `"FINISH"`). Each specialist node loops back to the supervisor, creating a dynamic multi-turn cycle that terminates only when the supervisor returns `FINISH`.

### `interrupt` and `Command(resume=…)`

The human-in-the-loop pattern calls `interrupt(payload)` inside `approval_node`. This suspends the graph and surfaces the payload to the caller. A `MemorySaver` checkpointer persists the suspended state across invocations. Resuming requires a second `.invoke(Command(resume={"approved": True/False}), config=config)` call with the same `thread_id`; LangGraph restores the checkpoint and continues from the suspension point.
