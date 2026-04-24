"""Sample reviewer — evaluates trajectories for quality issues."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.schema import (
    Sample,
    ReviewStep,
    ReviewVerdict,
    ReviewIssue,
    StepType,
)
from app.generators.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


@dataclass
class RCAAnalysis:
    """Root cause analysis produced by the reviewer."""
    causes: list[str]
    recommendations: list[str]


class SampleReviewer:
    """Reviews coding-agent trajectories and produces structured review steps.

    The reviewer identifies issues such as missing read-before-edit patterns,
    unsupported success claims, weak validation, and incorrect state transitions.
    """

    def __init__(self, prompt_builder: PromptBuilder | None = None) -> None:
        self.prompt_builder = prompt_builder or PromptBuilder()

    def review(self, sample: Sample) -> ReviewStep:
        """Evaluate a sample and return a structured review step."""
        issues = self._find_issues(sample)
        verdict = self._compute_verdict(issues)

        return ReviewStep(
            reviewer_role="senior_dataset_reviewer",
            verdict=verdict,
            issues=issues,
            recommended_next_step=self._recommended_next_step(verdict, issues),
        )

    def explain_rca(self, sample: Sample, review: ReviewStep) -> RCAAnalysis:
        """Explain root causes for the given review issues.

        In production this would call an LLM. Here we use heuristic rules.
        """
        causes: list[str] = []
        recommendations: list[str] = []

        for issue in review.issues:
            if "read_before_edit" in issue.message.lower():
                causes.append("No read_file call before the first edit_file call")
                recommendations.append("Add a read_file call to inspect the target file before editing")
            elif "success" in issue.message.lower() and "without" in issue.message.lower():
                causes.append("Final response claims success but no passing tool_result exists")
                recommendations.append("Ensure verification steps produce a successful tool_result")
            elif "grounded" in issue.message.lower():
                causes.append("Final response does not reference successful call_ids")
                recommendations.append("Update grounded_in to reference confirmed successful tool results")
            elif "redundant" in issue.message.lower():
                causes.append("Consecutive identical tool calls indicate lack of progress")
                recommendations.append("Remove duplicate calls or add intervening reasoning")
            elif "state" in issue.message.lower():
                causes.append("State transitions are inconsistent across the trace")
                recommendations.append("Ensure state_update.current_phase progresses logically")
            else:
                causes.append(f"Issue: {issue.message}")
                recommendations.append("Investigate and regenerate the faulty step")

        return RCAAnalysis(causes=causes, recommendations=recommendations)

    # ─── Private helpers ────────────────────────────────────────────────────────

    def _find_issues(self, sample: Sample) -> list[ReviewIssue]:
        """Heuristic issue detection. In production this uses an LLM."""
        issues: list[ReviewIssue] = []

        # Check read-before-edit
        has_read = any(
            s.type == StepType.TOOL_CALL and s.tool_name == "read_file"
            for s in sample.assistant_trace
        )
        has_edit = any(
            s.type == StepType.TOOL_CALL and s.tool_name == "edit_file"
            for s in sample.assistant_trace
        )
        if has_edit and not has_read:
            issues.append(ReviewIssue(
                severity="high",
                message="edit_file called without any prior read_file call",
                step_ref=None,
            ))

        # Check success claim without verification
        success_keywords = ("fixed", "done", "all good", "success")
        if sample.final_response.strip().lower().startswith(success_keywords):
            passing = [
                s for s in sample.get_tool_results()
                if s.status == "success"
            ]
            if not passing:
                issues.append(ReviewIssue(
                    severity="critical",
                    message="final response claims success without any passing test_result",
                    step_ref=None,
                ))

        # Check final grounding
        final = sample.get_final_answer_step()
        if final and final.grounded_in:
            successful_ids = {
                s.call_id for s in sample.get_tool_results()
                if s.status == "success"
            }
            if not any(cid in successful_ids for cid in final.grounded_in):
                issues.append(ReviewIssue(
                    severity="high",
                    message="grounded_in references call_ids but none are from successful results",
                    step_ref=final.call_id if hasattr(final, "call_id") else None,
                ))

        # Check for redundant consecutive calls
        prev_call: tuple[str, dict] | None = None
        for step in sample.assistant_trace:
            if step.type == StepType.TOOL_CALL:
                key = (step.tool_name, str(sorted(step.arguments.items())))
                if prev_call and prev_call[0] == key:
                    issues.append(ReviewIssue(
                        severity="medium",
                        message=f"consecutive duplicate call: {step.tool_name}",
                        step_ref=step.call_id,
                    ))
                prev_call = key
            else:
                prev_call = None

        return issues

    def _compute_verdict(self, issues: list[ReviewIssue]) -> ReviewVerdict:
        if not issues:
            return ReviewVerdict.APPROVED
        if any(i.severity == "critical" for i in issues):
            return ReviewVerdict.REJECTED
        return ReviewVerdict.NEEDS_FIX

    def _recommended_next_step(
        self,
        verdict: ReviewVerdict,
        issues: list[ReviewIssue],
    ) -> str:
        if verdict == ReviewVerdict.APPROVED:
            return "Sample is approved — ready for judge evaluation"
        if verdict == ReviewVerdict.REJECTED:
            return "Critical issues found — repair needed before re-review"
        return "Medium/high issues found — expand validation and regenerate faulty steps"
