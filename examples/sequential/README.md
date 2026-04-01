# Sequential Pattern

A linear two-node pipeline: `ResearchAgent → AnalystAgent → END`.

## When to use

Use the sequential pattern when tasks have strict data dependencies — the output of one step is the required input of the next. Common scenarios:

- Research-then-analysis pipelines where the analyst needs the full research brief before it can start
- Extract-then-transform workflows (e.g. parse a document, then classify its contents)
- Draft-then-review loops where a reviewer requires a complete draft
- Any workflow where parallelism would produce incorrect results due to ordering constraints

Avoid this pattern when steps are independent: in that case the parallel pattern will give you the same results in less wall-clock time.

## Graph topology

```
research_node  →  analyze_node  →  END
```

Both nodes share `SequentialState`. The research node writes `research_output`; the analyze node reads it and writes `analysis_output`. The `messages` field accumulates the full conversation log via the `operator.add` reducer.

## Running

```bash
uv run python examples/sequential/graph.py
```

## Expected output

```
Query: What are the benefits of microservices architecture?
------------------------------------------------------------
=== RESEARCH OUTPUT ===
**Key Concepts**
Microservices architecture decomposes an application into small, independently
deployable services that communicate over well-defined APIs...

[structured research brief with key concepts, current state, challenges, examples]

=== ANALYSIS OUTPUT ===
**Key Insights**
1. Independent deployability reduces release risk and accelerates delivery cadence...
2. Technology heterogeneity enables teams to choose the right tool per service...

[strategic analysis with insights, trade-offs, recommendations, open questions]
```

The research output is a structured information brief; the analysis output builds directly on it with strategic recommendations and trade-offs.
