# Supervisor Pattern

A router agent inspects each query and delegates to one of three specialist agents (research, code, data). After the specialist responds, control returns to the supervisor, which either routes to another specialist or terminates with `FINISH`.

## When to use

Use the supervisor pattern for multi-domain assistants where the appropriate specialist is not known upfront:

- General-purpose chatbots that must handle factual questions, coding tasks, and data analysis in the same session
- Agentic pipelines where follow-up steps depend on what the previous specialist returned
- Workflows that may require chaining several specialists (e.g. research a topic, then write code based on the findings)
- Any scenario where a single static routing table is insufficient and the routing decision requires reasoning over prior context

Avoid this pattern for workloads with a fixed, predictable sequence — sequential or parallel patterns are simpler and cheaper.

## Graph topology

```
START
  |
supervisor_node  <──────────────────┐
  |                                 │
  ├── "research" ──► research_node ─┤
  ├── "code"     ──► code_node     ─┤
  ├── "data"     ──► data_node     ─┤
  └── "FINISH"   ──► END
```

`supervisor_node` calls the LLM with a JSON routing prompt and writes the decision to `SupervisorState.next_agent`. `add_conditional_edges` reads that field and dispatches to the correct specialist. Each specialist appends its output to `messages` and loops back to the supervisor.

## Running

```bash
uv run python examples/supervisor/graph.py
```

## Expected output

The supervisor routes each query to the appropriate specialist. After the specialist responds, the supervisor terminates with `FINISH`. The "Routed to" line shows the last specialist that handled the query.

```
Query: Explain how transformer attention mechanisms work
------------------------------------------------------------
Routed to: FINISH

=== AGENT OUTPUT ===
Transformer attention uses scaled dot-product attention to compute a weighted
combination of value vectors...

Query: Write a Python function that computes the Fibonacci sequence iteratively
------------------------------------------------------------
Routed to: FINISH

=== AGENT OUTPUT ===
def fibonacci(n: int) -> list[int]:
    """Return the first n Fibonacci numbers iteratively."""
    ...

Query: What SQL query would I use to find the top 5 customers by total order value?
------------------------------------------------------------
Routed to: FINISH

=== AGENT OUTPUT ===
SELECT customer_id, SUM(order_value) AS total_value
FROM orders
GROUP BY customer_id
ORDER BY total_value DESC
LIMIT 5;
```

The supervisor correctly routes each query to a different specialist without being explicitly programmed with per-query rules.
