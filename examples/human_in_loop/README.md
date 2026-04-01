# Human-in-the-Loop Pattern

The agent proposes an action, then pauses via `interrupt()` and waits for explicit human approval before executing. A `MemorySaver` checkpointer persists the suspended state between the two `.invoke()` calls.

## When to use

Use this pattern whenever an agent might take an irreversible or high-impact action:

- Deleting or mutating database records
- Sending emails or notifications on behalf of a user
- Deploying infrastructure changes (Terraform apply, kubectl apply)
- Placing orders, making payments, or any financial transaction
- Calling external APIs with side effects that cannot be rolled back

The pattern enforces a mandatory human review gate: the agent cannot proceed without an explicit `Command(resume={"approved": True})`. Rejected actions are recorded without any side effects being applied.

## Graph topology

```
START
  |
plan_node        (LLM proposes a concrete action)
  |
approval_node    (interrupt() — execution pauses here)
  |
  ├── approved=True  ──► execute_node ──► END
  └── approved=False ──► reject_node  ──► END
```

`interrupt(payload)` inside `approval_node` suspends the graph and returns `payload` to the caller. The graph is resumed by calling `.invoke(Command(resume={"approved": bool}), config=same_config)`. LangGraph restores the checkpoint from `MemorySaver` and continues from the suspension point.

## Running

```bash
uv run python examples/human_in_loop/graph.py
```

## Expected output

```
Query: Delete all records from the 'test_users' table that were created before 2023-01-01
------------------------------------------------------------

[Phase 1] Running graph until human approval is required...

[INTERRUPT] Approval required:
  Proposed action: Execute a DELETE statement against the 'test_users' table:
  DELETE FROM test_users WHERE created_at < '2023-01-01';
  This will permanently remove approximately N rows...

[Phase 2] Simulating human APPROVAL (approved=True)...

=== EXECUTION RESULT (approved) ===
Action executed successfully. Deleted 142 records from 'test_users'
where created_at < '2023-01-01'. Transaction ID: txn-20240101-abc123.
No foreign key violations encountered.

============================================================
[Phase 3] Demonstrating REJECTION path...

=== EXECUTION RESULT (rejected) ===
Action rejected by human operator. No changes were made to any system.
```

The script runs the graph twice using different `thread_id` values to demonstrate both the approval and rejection paths independently.
