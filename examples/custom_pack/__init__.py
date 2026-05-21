"""Backward-compat re-export — prefer domain_packs.summariser."""

from domain_packs.summariser.pack import SummariserPack
from domain_packs.summariser.schemas import SummaryInput, SummaryOutput

__all__ = ["SummariserPack", "SummaryInput", "SummaryOutput"]
