"""Judge model — scores and accepts/rejects samples based on the quality rubric."""

from __future__ import annotations

import logging

from app.schema import (
    Sample,
    Quality,
    QualityDimension,
    JudgeVerdict,
    StepType,
)
from app.generators.pipeline_config import JudgeConfig
from app.validators.rules import acceptance_check

logger = logging.getLogger(__name__)


class Judge:
    """Scores each sample on 10 dimensions and issues accept/reject/needs_fix verdicts.

    In production, scoring is done by an LLM judge. Here we implement the heuristic
    rules plus a hook for LLM-based scoring.
    """

    def __init__(
        self,
        min_avg_score: float = 4.5,
        min_dimension_score: int = 4,
        config: JudgeConfig | None = None,
    ) -> None:
        self.min_avg_score = min_avg_score
        self.min_dimension_score = min_dimension_score
        self.cfg = config or JudgeConfig(
            min_avg_score=min_avg_score,
            min_dimension_score=min_dimension_score,
        )

    def judge(self, sample: Sample) -> tuple[JudgeVerdict, Quality]:
        """Score a sample and return a verdict and Quality object."""
        dimensions = self._score_dimensions(sample)
        score = dimensions.average
        issues: list[str] = []

        # Check dimension minimum
        min_dim = dimensions.min_score
        if min_dim < self.min_dimension_score:
            issues.append(f"dimension below {self.min_dimension_score}: min={min_dim}")

        # Check average
        if score < self.min_avg_score:
            issues.append(f"average below {self.min_avg_score}: avg={score:.2f}")

        # Check behavioral rules
        passed, rule_issues = acceptance_check(sample)
        issues.extend(rule_issues)

        if not issues:
            verdict = JudgeVerdict.ACCEPT
            reasoning = f"Accept — all dimensions >= {self.min_dimension_score}, avg={score:.2f}"
        elif len(issues) <= 2 and min_dim >= 3:
            verdict = JudgeVerdict.NEEDS_FIX
            reasoning = f"Needs fix: {'; '.join(issues)}"
        else:
            verdict = JudgeVerdict.REJECT
            reasoning = f"Reject: {'; '.join(issues)}"

        quality = Quality(
            score=score,
            judge_verdict=verdict,
            dimensions=dimensions,
            reasoning=reasoning,
        )

        return verdict, quality

    def _score_dimensions(self, sample: Sample) -> QualityDimension:
        """Score each of the 10 quality dimensions.

        In production, delegate to an LLM judge. Here use heuristics.
        """
        trace = sample.assistant_trace
        tool_calls = [s for s in trace if s.type == StepType.TOOL_CALL]
        tool_results = [s for s in trace if s.type == StepType.TOOL_RESULT]
        reasoning_steps = [s for s in trace if s.type == StepType.REASONING]
        passing = [r for r in tool_results if r.status == "success"]  # type: ignore[union]
        failing = [r for r in tool_results if r.status == "failure"]  # type: ignore[union]
        final = sample.get_final_answer_step()

        # 1. Task understanding
        task_understanding = self._score_task_understanding(sample)

        # 2. Reasoning quality
        reasoning_quality = min(5, 3 + len(reasoning_steps)) if reasoning_steps else 2

        # 3. Tool selection quality
        tool_selection = 5
        available = {t.name for t in sample.available_tools}
        if not available:
            available = {"read_file", "edit_file", "test_code"}
        for call in tool_calls:
            if call.tool_name not in available:
                tool_selection = max(2, tool_selection - 2)

        # 4. Tool call correctness
        tool_call_correctness = 5
        if len(tool_calls) != len(tool_results):
            tool_call_correctness -= 2
        if len(tool_results) == 0:
            tool_call_correctness = 1

        # 5. Tool result completeness
        tool_result_completeness = 5
        for r in tool_results:
            if not r.summary:
                tool_result_completeness -= 1
            if r.status == "failure" and not r.error:
                tool_result_completeness -= 1

        # 6. Validation correctness
        validation_correctness = 5
        for r in tool_results:
            if not r.validation.passed and r.status == "success":
                validation_correctness = 1
            if r.validation.passed and r.status == "failure":
                validation_correctness = 2

        # 7. State transition consistency
        state_transition_consistency = 5
        phases = [
            r.state_update.current_phase  # type: ignore[union]
            for r in tool_results
            if r.state_update and r.state_update.current_phase  # type: ignore[union]
        ]
        if phases:
            if phases[0] == "verification_complete" and "debugging" in phases[1:]:
                state_transition_consistency -= 2

        # 8. Minimal patch discipline
        edit_calls = [c for c in tool_calls if c.tool_name == "edit_file"]
        if not edit_calls:
            minimal_patch_discipline = 5
        elif len(edit_calls) == 1:
            minimal_patch_discipline = 4
        else:
            minimal_patch_discipline = min(5, 3 + len(edit_calls))

        # 9. Failure recovery quality
        failure_recovery_quality = 5
        if failing and not passing:
            failure_recovery_quality = 1
        elif failing and len(passing) < len(failing):
            failure_recovery_quality = 3

        # 10. Final response honesty
        final_response_honesty = self._score_final_honesty(sample, final, passing)

        return QualityDimension(
            task_understanding=task_understanding,
            reasoning_quality=reasoning_quality,
            tool_selection_quality=tool_selection,
            tool_call_correctness=tool_call_correctness,
            tool_result_completeness=tool_result_completeness,
            validation_correctness=validation_correctness,
            state_transition_consistency=state_transition_consistency,
            minimal_patch_discipline=minimal_patch_discipline,
            failure_recovery_quality=failure_recovery_quality,
            final_response_honesty=final_response_honesty,
        )

    def _score_task_understanding(self, sample: Sample) -> int:
        """Score how well the agent understood the task from user_input."""
        words_in_task = len(sample.user_input.split())
        has_trace = len(sample.assistant_trace) > 0
        if words_in_task < 5 or not has_trace:
            return 2
        if words_in_task > 20 and has_trace:
            return 5
        return 4

    def _score_final_honesty(
        self,
        sample: Sample,
        final: StepType | None,
        passing: list[StepType],
    ) -> int:
        """Score whether the final response honestly reflects the results."""
        if not sample.final_response.strip():
            return 1
        if sample.final_response.strip().lower().startswith(("fixed", "done", "all good")):
            if passing:
                return 5
            return 1
        return 4
