# Parallel Pattern

Three analyst agents (Technology, Market, Risk) run concurrently via the LangGraph `Send` API, then a consolidation node merges their results into a single executive report.

## When to use

Use the parallel pattern when multiple independent analyses of the same input can proceed simultaneously:

- Multi-perspective reports that need a technology view, a business view, and a risk view all at once
- Ensemble scoring where several models or personas evaluate the same document independently
- Parallel data enrichment where the same record is annotated by several classifiers
- Any fan-out workload where branches share only the input, not intermediate state

The key constraint is that parallel branches must be truly independent: no branch should read another branch's output. If ordering matters, use the sequential pattern instead.

## Graph topology

```
START
  |
fan_out_node          (emits one Send per analyst role)
  /       |       \
tech   market   risk  (all run concurrently via Send API)
  \       |       /
 consolidate_node     (merges all three analyses)
  |
 END
```

`fan_out_node` returns `[Send("analyst_node", branch) for branch in branches]`. LangGraph schedules all three `analyst_node` executions concurrently. Their `{"analyses": [...]}` partial updates are merged by the `operator.add` reducer on `ParallelState.analyses` before `consolidate_node` receives the full list.

## Running

```bash
uv run python examples/parallel/graph.py
```

## Expected output

```
Query: What are the trade-offs of adopting a microservices architecture?
------------------------------------------------------------------------
=== INDIVIDUAL ANALYSES ===
## Technology Analyst
**Technical Feasibility**
Microservices impose significant infrastructure complexity: service discovery,
distributed tracing, container orchestration...

## Market Analyst
**Market Adoption**
Microservices have reached mainstream adoption across cloud-native organisations.
Major hyperscalers (Netflix, Amazon, Uber) have publicly documented...

## Risk Analyst
**Operational Risks**
Network latency between services introduces failure modes absent in monoliths.
Distributed transactions require careful saga or two-phase commit patterns...

=== CONSOLIDATED REPORT ===
**Executive Summary**
Across the three analyses, a clear consensus emerges: microservices deliver
meaningful agility and scalability benefits at the cost of substantial
operational overhead...

[cross-cutting themes, reconciled trade-offs, prioritised recommendations]
```

All three analyst sections appear before the consolidated report, demonstrating that the branches ran independently and were merged before synthesis.
