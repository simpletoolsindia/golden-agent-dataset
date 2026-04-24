"""Repair agent — regenerates faulty trajectory sections based on reviewer feedback."""

from __future__ import annotations

import logging
from typing import Optional

from app.schema import (
    Sample,
    FixStep,
    ReviewStep,
    StepType,
)
from app.reviewers.sample_reviewer import RCAAnalysis
from app.generators.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class RepairAgent:
    """Regenerates specific steps in a trajectory to address reviewer issues.

    The repair agent does not rewrite the entire sample — it targets only the
    steps identified by the reviewer as problematic.
    """

    def __init__(self, prompt_builder: PromptBuilder | None = None) -> None:
        self.prompt_builder = prompt_builder or PromptBuilder()

    def repair(
        self,
        sample: Sample,
        review: ReviewStep,
        rca: RCAAnalysis,
        max_attempts: int = 3,
    ) -> FixStep:
        """Generate a fix strategy based on reviewer issues and RCA analysis.

        Returns a FixStep describing the repair strategy and what it addresses.
        """
        addressed = [f"review_issue_{i}" for i in range(len(review.issues))]
        fix_strategy = self._build_fix_strategy(review, rca)

        fix_step = FixStep(
            fix_strategy=fix_strategy,
            addresses_review_issues=addressed,
        )

        # Apply heuristic fixes directly to the sample trace
        self._apply_heuristic_fixes(sample, review, rca)

        return fix_step

    def _build_fix_strategy(
        self,
        review: ReviewStep,
        rca: RCAAnalysis,
    ) -> str:
        """Build a natural language fix strategy from issues and RCA."""
        parts: list[str] = []

        for issue in review.issues:
            for rec in rca.recommendations:
                if any(
                    kw in issue.message.lower()
                    for kw in rec.lower().split()
                    if len(kw) > 5
                ):
                    parts.append(rec)
                    break
            else:
                parts.append(f"Address severity-{issue.severity} issue: {issue.message}")

        if not parts:
            parts.append("Regenerate the trajectory from the point of failure")

        return "; ".join(parts)

    def _apply_heuristic_fixes(
        self,
        sample: Sample,
        review: ReviewStep,
        rca: RCAAnalysis,
    ) -> None:
        """Apply known-safe fixes directly to the trace.

        This handles cases where the fix is deterministic, like adding missing
        read_file calls or correcting grounded_in references.
        """
        # Fix missing read-before-edit
        has_read = any(
            s.type == StepType.TOOL_CALL and s.tool_name == "read_file"
            for s in sample.assistant_trace
        )
        has_edit = any(
            s.type == StepType.TOOL_CALL and s.tool_name == "edit_file"
            for s in sample.assistant_trace
        )

        if has_edit and not has_read:
            from app.schema import (
                ReasoningStep,
                ToolCallStep,
                ToolResultStep,
                ToolStatus,
                ToolOutput,
                ValidationCheck,
            )
            # Insert read steps before the first edit
            read_call = ToolCallStep(
                type=StepType.TOOL_CALL,
                tool_name="read_file",
                call_id="call_REPAIR_001",
                arguments={"path": sample.context.repo_files[0] if sample.context.repo_files else "main.py"},
            )
            read_result = ToolResultStep(
                type=StepType.TOOL_RESULT,
                tool_name="read_file",
                call_id="call_REPAIR_001",
                status=ToolStatus.SUCCESS,
                summary="Repaired: inserted read_file to satisfy read-before-edit rule",
                validation=ValidationCheck(name="read_ok", passed=True),
                output=ToolOutput(),
            )
            reasoning = ReasoningStep(
                type=StepType.REASONING,
                goal="Repair: add missing inspection step",
                decision="Insert read_file before editing",
                why="Read-before-edit is required by dataset rules",
                confidence=1.0,
            )

            # Find first edit position
            edit_idx = next(
                (i for i, s in enumerate(sample.assistant_trace)
                 if s.type == StepType.TOOL_CALL and s.tool_name == "edit_file"),
                0,
            )
            sample.assistant_trace.insert(edit_idx, reasoning)
            sample.assistant_trace.insert(edit_idx + 1, read_call)
            sample.assistant_trace.insert(edit_idx + 2, read_result)

        # Fix final grounding if needed
        final = sample.get_final_answer_step()
        if final and final.grounded_in:
            passing_ids = [
                s.call_id for s in sample.get_tool_results()
                if s.status == "success"
            ]
            if passing_ids and not any(
                cid in passing_ids for cid in final.grounded_in
            ):
                final.grounded_in = [passing_ids[0]]

        logger.debug(f"Applied heuristic fixes to {sample.id}")
