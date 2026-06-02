"""Built-in domain pack registration (single source of truth)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pack_kernel.base_pack import BaseDomainPack


def all_builtin_pack_classes() -> list[type[BaseDomainPack]]:
    """Import and return all built-in pack classes (lazy to avoid cycles)."""
    from domain_packs.finance.financial_memo.pack import FinancialMemoPack
    from domain_packs.hr.hr_policy_qa.pack import HrPolicyQaPack
    from domain_packs.hr.job_description_writer.pack import JobDescriptionWriterPack
    from domain_packs.hr.talent_screening.pack import TalentScreeningPack
    from domain_packs.legal.contract_reviewer.pack import ContractReviewerPack
    from domain_packs.productivity.executive_brief.pack import ExecutiveBriefPack
    from domain_packs.productivity.meeting_prep.pack import MeetingPrepPack
    from domain_packs.productivity.rfp_assistant.pack import RfpAssistantPack
    from domain_packs.productivity.summariser.pack import SummariserPack
    from domain_packs.productivity.support_triage.pack import SupportTriagePack
    from domain_packs.research.analysis_only.pack import AnalysisOnlyPack
    from domain_packs.research.research_analysis.pack import ResearchAnalysisPack
    from domain_packs.research.research_only.pack import ResearchOnlyPack

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
    from pack_kernel.registry import PackRegistry

    for pack_cls in all_builtin_pack_classes():
        PackRegistry.register(pack_cls)
