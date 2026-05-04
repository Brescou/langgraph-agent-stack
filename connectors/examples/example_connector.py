"""
example_connector.py — In-memory stub illustrating the connector contract.

Not for production. Domain packs would inject a real connector instance or
factory when implementing retrieval-augmented behaviour.
"""

from __future__ import annotations

from typing import Any, ClassVar

from connectors.base import BaseConnector, ConnectorRequest, ConnectorResult


class ExampleMemoryConnector(BaseConnector):
    """Returns canned rows matching a substring in ``query`` (demo only)."""

    connector_id: ClassVar[str] = "example_memory"
    name: ClassVar[str] = "Example in-memory connector"
    description: ClassVar[str] = (
        "Returns static snippets when the query contains the word 'demo'."
    )

    async def fetch(self, request: ConnectorRequest) -> ConnectorResult:
        _ = request.filters  # reserved for future filtering
        if "demo" not in request.query.lower():
            return ConnectorResult(records=(), metadata={"matched": False})
        rows: tuple[dict[str, Any], ...] = (
            {
                "source": "example",
                "snippet": "This is a canned result for demonstration.",
                "score": 1.0,
            },
        )
        return ConnectorResult(
            records=rows,
            metadata={"matched": True, "limit": request.limit},
        )
