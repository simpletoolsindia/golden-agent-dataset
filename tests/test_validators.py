"""Unit tests for validators."""

import pytest

from app.schema import (
    Sample,
    StepType,
    ReasoningStep,
    DecisionStep,
    ToolCallStep,
    ToolResultStep,
    FinalStep,
    ToolStatus,
    ToolOutput,
    ValidationCheck,
    Quality,
    QualityDimension,
    TaskCategory,
    Language,
    ToolDefinition,
    JudgeVerdict,
)
from app.validators import (
    run_all_validators,
    run_structural_validators,
    run_behavioral_validators,
    ValidationIssue,
)
from app.validators.rules import acceptance_check


def make_minimal_sample(**kwargs) -> Sample:
    defaults = dict(
        id="test_sample",
        category=TaskCategory.BUG_FIX,
        language=Language.PYTHON,
        user_input="Fix the bug",
        assistant_trace=[],
    )
    defaults.update(kwargs)
    return Sample(**defaults)


class TestStructuralValidators:
    def test_valid_sample_passes(self):
        sample = make_minimal_sample(
            assistant_trace=[
                ToolCallStep(
                    tool_name="read_file",
                    call_id="call_0001",
                    arguments={"path": "main.py"},
                ),
                ToolResultStep(
                    tool_name="read_file",
                    call_id="call_0001",
                    status=ToolStatus.SUCCESS,
                    summary="ok",
                    validation=ValidationCheck(name="ok", passed=True),
                    output=ToolOutput(),
                ),
            ],
        )
        issues = run_structural_validators(sample)
        assert issues == []

    def test_orphan_result_detected(self):
        sample = make_minimal_sample(
            assistant_trace=[
                ToolResultStep(
                    tool_name="read_file",
                    call_id="call_0001",
                    status=ToolStatus.SUCCESS,
                    summary="orphan",
                    validation=ValidationCheck(name="ok", passed=True),
                    output=ToolOutput(),
                ),
            ],
        )
        issues = run_structural_validators(sample)
        assert any(
            i.validator == "orphan_result" for i in issues
        )

    def test_call_without_result_detected(self):
        sample = make_minimal_sample(
            assistant_trace=[
                ToolCallStep(
                    tool_name="read_file",
                    call_id="call_0001",
                    arguments={"path": "main.py"},
                ),
            ],
        )
        issues = run_structural_validators(sample)
        assert any(
            i.validator == "call_result_matching" for i in issues
        )


class TestBehavioralValidators:
    def test_read_before_edit_violation(self):
        sample = make_minimal_sample(
            available_tools=[
                ToolDefinition(name="read_file", description="Read a file"),
                ToolDefinition(name="edit_file", description="Edit a file"),
            ],
            assistant_trace=[
                ToolCallStep(
                    tool_name="edit_file",
                    call_id="call_0001",
                    arguments={"path": "main.py", "old": "x", "new": "y"},
                ),
            ],
        )
        issues = run_behavioral_validators(sample)
        assert any(
            "read_before_edit" in i.validator
            for i in issues
        )

    def test_read_before_edit_ok(self):
        sample = make_minimal_sample(
            available_tools=[
                ToolDefinition(name="read_file", description="Read a file"),
                ToolDefinition(name="edit_file", description="Edit a file"),
            ],
            assistant_trace=[
                ToolCallStep(
                    tool_name="read_file",
                    call_id="call_0001",
                    arguments={"path": "main.py"},
                ),
                ToolResultStep(
                    tool_name="read_file",
                    call_id="call_0001",
                    status=ToolStatus.SUCCESS,
                    summary="ok",
                    validation=ValidationCheck(name="ok", passed=True),
                    output=ToolOutput(),
                ),
                ToolCallStep(
                    tool_name="edit_file",
                    call_id="call_0002",
                    arguments={"path": "main.py", "old": "x", "new": "y"},
                ),
            ],
        )
        issues = run_behavioral_validators(sample)
        assert not any(
            "read_before_edit" in i.validator
            for i in issues
        )

    def test_tool_availability_violation(self):
        sample = make_minimal_sample(
            available_tools=[
                ToolDefinition(name="read_file", description="Read a file"),
            ],
            assistant_trace=[
                ToolCallStep(
                    tool_name="edit_file",
                    call_id="call_0001",
                    arguments={"path": "main.py", "old": "x", "new": "y"},
                ),
            ],
        )
        issues = run_behavioral_validators(sample)
        assert any(
            "tool_availability" in i.validator
            for i in issues
        )


class TestValidationValidators:
    def test_success_claim_without_verification(self):
        sample = make_minimal_sample(
            final_response="Fixed the bug",
            assistant_trace=[
                ToolCallStep(
                    tool_name="read_file",
                    call_id="call_0001",
                    arguments={"path": "main.py"},
                ),
            ],
        )
        issues = run_all_validators(sample)
        assert any(
            "success_claims" in i.validator
            for i in issues
        )

    def test_success_claim_with_verification_passes(self):
        sample = make_minimal_sample(
            final_response="Fixed the bug",
            assistant_trace=[
                ToolResultStep(
                    tool_name="test_code",
                    call_id="call_0001",
                    status=ToolStatus.SUCCESS,
                    summary="3 tests passed",
                    validation=ValidationCheck(name="ok", passed=True),
                    output=ToolOutput(),
                ),
                FinalStep(
                    content="Fixed the bug, all tests pass",
                    grounded_in=["call_0001"],
                ),
            ],
        )
        issues = run_all_validators(sample)
        assert not any(
            "success_claims" in i.validator
            for i in issues
        )


class TestAcceptanceCheck:
    def test_accepts_high_quality_sample(self):
        sample = make_minimal_sample(
            assistant_trace=[
                ToolCallStep(
                    tool_name="read_file",
                    call_id="call_0001",
                    arguments={"path": "main.py"},
                ),
                ToolResultStep(
                    tool_name="read_file",
                    call_id="call_0001",
                    status=ToolStatus.SUCCESS,
                    summary="ok",
                    validation=ValidationCheck(name="ok", passed=True),
                    output=ToolOutput(),
                ),
                FinalStep(
                    content="Fixed the bug",
                    grounded_in=["call_0001"],
                ),
            ],
            final_response="Fixed the bug",
            quality=Quality(
                score=4.9,
                judge_verdict=JudgeVerdict.ACCEPT,
                dimensions=QualityDimension(
                    task_understanding=5,
                    reasoning_quality=5,
                    tool_selection_quality=5,
                    tool_call_correctness=5,
                    tool_result_completeness=5,
                    validation_correctness=5,
                    state_transition_consistency=5,
                    minimal_patch_discipline=5,
                    failure_recovery_quality=5,
                    final_response_honesty=5,
                ),
            ),
        )
        passed, issues = acceptance_check(sample)
        assert passed is True
        assert issues == []

    def test_rejects_empty_final_response(self):
        sample = make_minimal_sample(
            final_response="",
            assistant_trace=[],
            quality=Quality(
                score=4.5,
                judge_verdict=JudgeVerdict.ACCEPT,
                dimensions=QualityDimension(
                    task_understanding=5,
                    reasoning_quality=5,
                    tool_selection_quality=5,
                    tool_call_correctness=5,
                    tool_result_completeness=5,
                    validation_correctness=5,
                    state_transition_consistency=5,
                    minimal_patch_discipline=5,
                    failure_recovery_quality=5,
                    final_response_honesty=5,
                ),
            ),
        )
        passed, issues = acceptance_check(sample)
        assert passed is False
        assert any("empty" in i.lower() for i in issues)
