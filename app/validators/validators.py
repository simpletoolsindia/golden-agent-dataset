"""Structural, behavioral, and quality validators for golden dataset samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.schema import (
    Sample,
    StepType,
    ToolCallStep,
    ToolResultStep,
    FinalStep,
    Quality,
    JudgeVerdict,
)
from app.validators.rules import acceptance_check


@dataclass
class ValidationIssue:
    validator: str
    message: str
    severity: str = "error"

    def __str__(self) -> str:
        return f"[{self.validator}] {self.message}"


# ─── Structural validators ──────────────────────────────────────────────────


def validate_json_schema(sample: Sample) -> list[ValidationIssue]:
    """Sample must be valid JSON and conform to Pydantic schema."""
    issues: list[ValidationIssue] = []
    try:
        sample.model_validate(sample.model_dump())
    except Exception as e:
        issues.append(ValidationIssue("json_schema", f"validation failed: {e}"))
    return issues


def validate_required_fields(sample: Sample) -> list[ValidationIssue]:
    """All required top-level fields must be present and non-null."""
    issues: list[ValidationIssue] = []
    for field in ("id", "category", "language", "user_input"):
        if not getattr(sample, field, None):
            issues.append(ValidationIssue("required_fields", f"'{field}' is missing"))
    return issues


def validate_call_result_matching(sample: Sample) -> list[ValidationIssue]:
    """Every tool_call must have a matching tool_result with the same call_id."""
    issues: list[ValidationIssue] = []
    calls: dict[str, ToolCallStep] = {}
    results: dict[str, ToolResultStep] = {}

    for step in sample.assistant_trace:
        if step.type == StepType.TOOL_CALL:
            calls[step.call_id] = step  # type: ignore[assignment]
        elif step.type == StepType.TOOL_RESULT:
            results[step.call_id] = step  # type: ignore[assignment]

    for call_id, call in calls.items():
        if call_id not in results:
            issues.append(ValidationIssue(
                "call_result_matching",
                f"call '{call_id}' ({call.tool_name}) has no matching result",
            ))

    return issues


def validate_step_order(sample: Sample) -> list[ValidationIssue]:
    """tool_result must not appear before its corresponding tool_call."""
    issues: list[ValidationIssue] = []
    seen_calls: set[str] = set()

    for step in sample.assistant_trace:
        if step.type == StepType.TOOL_CALL:
            seen_calls.add(step.call_id)  # type: ignore[union]
        elif step.type == StepType.TOOL_RESULT:
            if step.call_id not in seen_calls:  # type: ignore[union]
                issues.append(ValidationIssue(
                    "step_order",
                    f"tool_result '{step.call_id}' appears before its tool_call",
                ))

    return issues


def validate_no_orphan_results(sample: Sample) -> list[ValidationIssue]:
    """tool_result with no corresponding tool_call is invalid."""
    issues: list[ValidationIssue] = []
    seen_calls: set[str] = set()

    for step in sample.assistant_trace:
        if step.type == StepType.TOOL_CALL:
            seen_calls.add(step.call_id)  # type: ignore[union]

    for step in sample.assistant_trace:
        if step.type == StepType.TOOL_RESULT and step.call_id not in seen_calls:  # type: ignore[union]
            issues.append(ValidationIssue(
                "orphan_result",
                f"tool_result call_id '{step.call_id}' has no matching tool_call",
            ))

    return issues


# ─── Behavioral validators ──────────────────────────────────────────────────


def validate_reasoning_supports_action(sample: Sample) -> list[ValidationIssue]:
    """Every decision/action should be preceded by a reasoning step."""
    issues: list[ValidationIssue] = []
    had_reasoning = False

    for step in sample.assistant_trace:
        if step.type == StepType.REASONING:
            had_reasoning = True
        elif step.type == StepType.DECISION and not had_reasoning:
            issues.append(ValidationIssue(
                "reasoning_support",
                "decision step without preceding reasoning",
            ))
            had_reasoning = False
        elif step.type in (StepType.TOOL_CALL, StepType.FIX):
            had_reasoning = False

    return issues


def validate_tool_availability(sample: Sample) -> list[ValidationIssue]:
    """Every tool_call must use only tools listed in available_tools."""
    issues: list[ValidationIssue] = []
    available = {t.name for t in sample.available_tools}

    for step in sample.assistant_trace:
        if step.type == StepType.TOOL_CALL:
            if step.tool_name not in available:  # type: ignore[union]
                issues.append(ValidationIssue(
                    "tool_availability",
                    f"tool '{step.tool_name}' is not in available_tools",
                ))

    return issues


def validate_read_before_edit(sample: Sample) -> list[ValidationIssue]:
    """If edit_file appears, at least one read_file must precede it."""
    issues: list[ValidationIssue] = []
    reads: set[str] = set()
    edit_appeared = False

    for step in sample.assistant_trace:
        if step.type == StepType.TOOL_CALL:
            name = step.tool_name  # type: ignore[union]
            if name == "read_file":
                reads.add(step.call_id)  # type: ignore[union]
            elif name == "edit_file":
                edit_appeared = True

    if edit_appeared and not reads:
        issues.append(ValidationIssue(
            "read_before_edit",
            "edit_file used without any prior read_file call",
        ))

    return issues


def validate_no_redundant_steps(sample: Sample) -> list[ValidationIssue]:
    """Consecutive identical tool_calls are redundant."""
    issues: list[ValidationIssue] = []
    prev: Optional[ToolCallStep] = None

    for step in sample.assistant_trace:
        if step.type == StepType.TOOL_CALL:
            if prev and prev.tool_name == step.tool_name and prev.arguments == step.arguments:  # type: ignore[union]
                issues.append(ValidationIssue(
                    "redundant_steps",
                    f"consecutive duplicate call: {step.tool_name} '{step.call_id}'",
                ))
            prev = step  # type: ignore[assignment]
        else:
            prev = None

    return issues


# ─── Validation validators ─────────────────────────────────────────────────────


def validate_success_claims(sample: Sample) -> list[ValidationIssue]:
    """Claims of success must be backed by a successful tool_result."""
    issues: list[ValidationIssue] = []

    if sample.final_response.strip().lower().startswith(("fixed", "done", "all good", "success")):
        successful_results = [
            s for s in sample.get_tool_results()
            if s.status == "success"
        ]
        if not successful_results:
            issues.append(ValidationIssue(
                "success_claims",
                "final response claims success but no successful tool_result exists",
            ))

    return issues


def validate_state_transitions(sample: Sample) -> list[ValidationIssue]:
    """State updates must be consistent across the trace."""
    issues: list[ValidationIssue] = []
    prev_phase: Optional[str] = None

    for step in sample.assistant_trace:
        if step.type == StepType.TOOL_RESULT and step.state_update:
            phase = step.state_update.current_phase  # type: ignore[union]
            if prev_phase == "verification_complete" and phase == "debugging":
                issues.append(ValidationIssue(
                    "state_transition",
                    "transitioned from verification_complete back to debugging",
                ))
            prev_phase = phase  # type: ignore[assignment]

    return issues


def validate_final_response_grounding(sample: Sample) -> list[ValidationIssue]:
    """Final response must reference call_ids that succeeded."""
    issues: list[ValidationIssue] = []
    final = sample.get_final_answer_step()

    if not final:
        if sample.final_response.strip():
            return []  # final_response text exists, that's fine
        issues.append(ValidationIssue(
            "final_grounding",
            "no final step in trace and final_response is empty",
        ))
        return issues

    if not final.grounded_in:
        issues.append(ValidationIssue(
            "final_grounding",
            "final step has no grounded_in call_ids",
        ))
        return issues

    successful_ids = {
        s.call_id for s in sample.get_tool_results()
        if s.status == "success"
    }
    grounded_success = [
        cid for cid in final.grounded_in
        if cid in successful_ids
    ]
    if not grounded_success:
        issues.append(ValidationIssue(
            "final_grounding",
            "grounded_in call_ids exist but none are from successful tool_results",
        ))

    return issues


def validate_quality_dimensions(sample: Sample) -> list[ValidationIssue]:
    """Quality scores must be within valid ranges."""
    issues: list[ValidationIssue] = []

    if not (0.0 <= sample.quality.score <= 5.0):
        issues.append(ValidationIssue(
            "quality_score_range",
            f"quality.score {sample.quality.score} is outside [0, 5]",
        ))

    return issues


def validate_judge_verdict_consistency(sample: Sample) -> list[ValidationIssue]:
    """Judge verdict must be consistent with quality scores."""
    issues: list[ValidationIssue] = []

    if sample.quality.judge_verdict == JudgeVerdict.ACCEPT:
        if sample.quality.dimensions.min_score < 4:
            issues.append(ValidationIssue(
                "judge_consistency",
                "verdict is 'accept' but a dimension scored below 4",
            ))
        if sample.quality.dimensions.average < 4.5:
            issues.append(ValidationIssue(
                "judge_consistency",
                "verdict is 'accept' but average dimension score is below 4.5",
            ))

    return issues


# ─── Aggregator ──────────────────────────────────────────────────────────────


ALL_STRUCTURAL_VALIDATORS = [
    validate_json_schema,
    validate_required_fields,
    validate_call_result_matching,
    validate_step_order,
    validate_no_orphan_results,
]

ALL_BEHAVIORAL_VALIDATORS = [
    validate_reasoning_supports_action,
    validate_tool_availability,
    validate_read_before_edit,
    validate_no_redundant_steps,
]

ALL_VALIDATION_VALIDATORS = [
    validate_success_claims,
    validate_state_transitions,
    validate_final_response_grounding,
    validate_quality_dimensions,
    validate_judge_verdict_consistency,
]


def run_all_validators(sample: Sample) -> list[ValidationIssue]:
    """Run all validator groups and return aggregated issues."""
    issues: list[ValidationIssue] = []

    for validator in (
        ALL_STRUCTURAL_VALIDATORS
        + ALL_BEHAVIORAL_VALIDATORS
        + ALL_VALIDATION_VALIDATORS
    ):
        issues.extend(validator(sample))

    return issues


def run_structural_validators(sample: Sample) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for v in ALL_STRUCTURAL_VALIDATORS:
        issues.extend(v(sample))
    return issues


def run_behavioral_validators(sample: Sample) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for v in ALL_BEHAVIORAL_VALIDATORS:
        issues.extend(v(sample))
    return issues
