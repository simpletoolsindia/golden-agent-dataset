"""Unit tests for the schema models."""

import pytest
from pydantic import ValidationError

from app.schema import (
    Sample,
    StepType,
    ReasoningStep,
    DecisionStep,
    ToolCallStep,
    ToolResultStep,
    ReviewStep,
    FixStep,
    FinalStep,
    ToolStatus,
    ToolOutput,
    ValidationCheck,
    StateUpdate,
    Quality,
    QualityDimension,
    TaskCategory,
    Language,
    Difficulty,
    ReviewVerdict,
    JudgeVerdict,
    ToolDefinition,
)


class TestSample:
    def test_minimal_sample(self):
        sample = Sample(
            id="sample_test_001",
            category=TaskCategory.BUG_FIX,
            language=Language.PYTHON,
            user_input="Fix the bug",
        )
        assert sample.id == "sample_test_001"
        assert sample.category == TaskCategory.BUG_FIX
        assert sample.difficulty == Difficulty.MEDIUM

    def test_sample_with_trace(self):
        reasoning = ReasoningStep(
            goal="Fix the bug",
            decision="Read the file first",
            why="Need to understand the current state",
            confidence=0.9,
        )
        call = ToolCallStep(
            tool_name="read_file",
            call_id="call_0001",
            arguments={"path": "main.py"},
        )
        result = ToolResultStep(
            tool_name="read_file",
            call_id="call_0001",
            status=ToolStatus.SUCCESS,
            summary="File read successfully",
            validation=ValidationCheck(name="ok", passed=True),
            output=ToolOutput(),
        )
        final = FinalStep(
            content="Fixed the bug by updating main.py",
            grounded_in=["call_0001"],
        )

        sample = Sample(
            id="sample_test_002",
            category=TaskCategory.BUG_FIX,
            language=Language.PYTHON,
            user_input="Fix the bug",
            assistant_trace=[reasoning, call, result, final],
        )

        assert len(sample.assistant_trace) == 4
        assert sample.get_call_ids() == {"call_0001"}
        assert len(sample.get_tool_results()) == 1
        assert sample.get_final_answer_step() == final

    def test_get_call_ids(self):
        sample = Sample(
            id="sample_test_003",
            category=TaskCategory.FEATURE_ADD,
            language=Language.PYTHON,
            user_input="Add a feature",
            assistant_trace=[
                ToolCallStep(
                    tool_name="read_file",
                    call_id="call_001",
                    arguments={"path": "a.py"},
                ),
                ToolCallStep(
                    tool_name="edit_file",
                    call_id="call_002",
                    arguments={"path": "a.py", "old": "x", "new": "y"},
                ),
            ],
        )
        assert sample.get_call_ids() == {"call_001", "call_002"}


class TestQualityDimension:
    def test_average(self):
        dim = QualityDimension(
            task_understanding=5,
            reasoning_quality=4,
            tool_selection_quality=5,
            tool_call_correctness=4,
            tool_result_completeness=5,
            validation_correctness=5,
            state_transition_consistency=5,
            minimal_patch_discipline=5,
            failure_recovery_quality=5,
            final_response_honesty=5,
        )
        assert dim.average == 4.8

    def test_min_score(self):
        dim = QualityDimension(
            task_understanding=5,
            reasoning_quality=2,  # lowest
            tool_selection_quality=4,
            tool_call_correctness=5,
            tool_result_completeness=5,
            validation_correctness=5,
            state_transition_consistency=5,
            minimal_patch_discipline=5,
            failure_recovery_quality=5,
            final_response_honesty=5,
        )
        assert dim.min_score == 2


class TestToolResultStep:
    def test_success_result(self):
        result = ToolResultStep(
            tool_name="test_code",
            call_id="call_0001",
            status=ToolStatus.SUCCESS,
            exit_code=0,
            summary="3 tests passed",
            validation=ValidationCheck(name="exit_code", passed=True),
            output=ToolOutput(stdout="=== 3 passed ==="),
        )
        assert result.status == ToolStatus.SUCCESS
        assert result.exit_code == 0

    def test_failure_result(self):
        result = ToolResultStep(
            tool_name="test_code",
            call_id="call_0002",
            status=ToolStatus.FAILURE,
            exit_code=1,
            summary="1 test failed",
            validation=ValidationCheck(name="exit_code", passed=False),
            output=ToolOutput(stdout="1 failed"),
            error={
                "category": "test_failure",
                "message": "AssertionError: expected READ_COMMITTED, got AUTOCOMMIT",
                "retryable": True,
            },
        )
        assert result.status == ToolStatus.FAILURE
        assert result.error is not None
        assert result.error["category"] == "test_failure"


class TestReviewStep:
    def test_approved_review(self):
        review = ReviewStep(
            verdict=ReviewVerdict.APPROVED,
            issues=[],
            summary="Clean trajectory",
        )
        assert review.verdict == ReviewVerdict.APPROVED
        assert len(review.issues) == 0

    def test_needs_fix_review(self):
        review = ReviewStep(
            verdict=ReviewVerdict.NEEDS_FIX,
            issues=[
                {
                    "severity": "high",
                    "message": "edit_file called without prior read_file",
                    "step_ref": "call_0001",
                }
            ],
            recommended_next_step="Add read_file before edit",
        )
        assert review.verdict == ReviewVerdict.NEEDS_FIX
        assert len(review.issues) == 1
        assert review.issues[0]["severity"] == "high"


class TestFixStep:
    def test_fix_step(self):
        fix = FixStep(
            fix_strategy="Add missing read_file call before the first edit",
            addresses_review_issues=["review_issue_0"],
        )
        assert len(fix.addresses_review_issues) == 1


class TestDecisionStep:
    def test_rejected_options(self):
        decision = DecisionStep(
            based_on=["call_0001"],
            decision="edit_file",
            why="session.py is the minimal change point",
            rejected_options=[
                "rewrite entire transaction manager",
                "modify tests first",
            ],
        )
        assert len(decision.rejected_options) == 2
