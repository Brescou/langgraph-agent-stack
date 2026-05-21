"""Built-in domain pack registration (single source of truth)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from platform.base_pack import BaseDomainPack


def all_builtin_pack_classes() -> list[type[BaseDomainPack]]:
    """Import and return all built-in pack classes (lazy to avoid cycles)."""
    from domain_packs.analysis_only.pack import AnalysisOnlyPack
    from domain_packs.contract_reviewer.pack import ContractReviewerPack
    from domain_packs.executive_brief.pack import ExecutiveBriefPack
    from domain_packs.financial_memo.pack import FinancialMemoPack
    from domain_packs.meeting_prep.pack import MeetingPrepPack
    from domain_packs.research_analysis.pack import ResearchAnalysisPack
    from domain_packs.research_only.pack import ResearchOnlyPack
    from domain_packs.rfp_assistant.pack import RfpAssistantPack
    from domain_packs.rh.hr_policy_qa.pack import HrPolicyQaPack
    from domain_packs.rh.job_description_writer.pack import JobDescriptionWriterPack
    from domain_packs.rh.talent_screening.pack import TalentScreeningPack
    from domain_packs.summariser.pack import SummariserPack
    from domain_packs.support_triage.pack import SupportTriagePack

    return [
        ResearchAnalysisPack,
        ResearchOnlyPack,
        SummariserPack,
        AnalysisOnlyPack,
        MeetingPrepPack,
        RfpAssistantPack,
        SupportTriagePack,
        ExecutiveBriefPack,
        ContractReviewerPack,
        FinancialMemoPack,
        TalentScreeningPack,
        JobDescriptionWriterPack,
        HrPolicyQaPack,
    ]


def register_builtin_packs() -> None:
    from platform.registry import PackRegistry

    for pack_cls in all_builtin_pack_classes():
        PackRegistry.register(pack_cls)
