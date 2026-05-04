"""
connectors — Optional adapters for retrieving external data (foundation only).

Import ``BaseConnector``, ``ConnectorRequest``, and ``ConnectorResult`` from
``connectors`` or ``connectors.base``. No side effects; nothing registers at import.
"""

from connectors.base import BaseConnector, ConnectorRequest, ConnectorResult

__all__ = ["BaseConnector", "ConnectorRequest", "ConnectorResult"]
