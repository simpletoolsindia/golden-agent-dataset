"""Unit tests for the generator, reviewer, judge, and exporter."""

import pytest
from pathlib import Path
import tempfile
import os

from app.schema import (
    Sample,
    StepType,
    ReasoningStep,
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
    Difficulty,
    JudgeVerdict,
    ToolDefinition,
)
from app.generators import TrajectoryGenerator, PipelineConfig
from app.reviewers import SampleReviewer
from app.repair import RepairAgent
from app.judge import Judge
from app.exporters import JSONLExporter, load_jsonl


class TestTrajectoryGenerator:
    def test_generate_one_returns_sample(self):
        cfg = PipelineConfig()
        gen = TrajectoryGenerator(cfg)
        sample = gen.generate_one()

        assert isinstance(sample, Sample)
        assert sample.id.startswith("sample_")
        assert len(sample.assistant_trace) > 0

    def test_generated_sample_has_quality_score(self):
        cfg = PipelineConfig()
        gen = TrajectoryGenerator(cfg)
        sample = gen.generate_one()

        assert sample.quality.score > 0
        assert sample.quality.judge_verdict in list(JudgeVerdict)


class TestSampleReviewer:
    def test_review_approves_clean_sample(self):
        reviewer = SampleReviewer()
        sample = Sample(
            id="test_review_001",
            category=TaskCategory.BUG_FIX,
            language=Language.PYTHON,
            user_input="Fix the bug",
            available_tools=[
                ToolDefinition(name="read_file", description="Read"),
                ToolDefinition(name="edit_file", description="Edit"),
            ],
            assistant_trace=[
                ReasoningStep(
                    goal="Fix the bug",
                    decision="Read the file first",
                    why="Need to understand current state",
                    confidence=0.9,
                ),
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
                ToolResultStep(
                    tool_name="edit_file",
                    call_id="call_0002",
                    status=ToolStatus.SUCCESS,
                    summary="ok",
                    validation=ValidationCheck(name="ok", passed=True),
                    output=ToolOutput(),
                ),
                ToolResultStep(
                    tool_name="test_code",
                    call_id="call_0003",
                    status=ToolStatus.SUCCESS,
                    exit_code=0,
                    summary="3 tests passed",
                    validation=ValidationCheck(name="ok", passed=True),
                    output=ToolOutput(),
                ),
                FinalStep(
                    content="Fixed the bug, tests pass",
                    grounded_in=["call_0003"],
                ),
            ],
            final_response="Fixed the bug, tests pass",
        )

        review = reviewer.review(sample)
        assert review.verdict.value == "approved"
        assert len(review.issues) == 0

    def test_review_detects_missing_read(self):
        reviewer = SampleReviewer()
        sample = Sample(
            id="test_review_002",
            category=TaskCategory.BUG_FIX,
            language=Language.PYTHON,
            user_input="Fix the bug",
            available_tools=[
                ToolDefinition(name="read_file", description="Read"),
                ToolDefinition(name="edit_file", description="Edit"),
            ],
            assistant_trace=[
                ToolCallStep(
                    tool_name="edit_file",
                    call_id="call_0001",
                    arguments={"path": "main.py", "old": "x", "new": "y"},
                ),
                ToolResultStep(
                    tool_name="edit_file",
                    call_id="call_0001",
                    status=ToolStatus.SUCCESS,
                    summary="ok",
                    validation=ValidationCheck(name="ok", passed=True),
                    output=ToolOutput(),
                ),
            ],
            final_response="Fixed the bug",
        )

        review = reviewer.review(sample)
        assert len(review.issues) > 0
        assert any("read" in i.message.lower() for i in review.issues)


class TestJudge:
    def test_judge_accepts_high_quality_sample(self):
        judge = Judge(min_avg_score=4.5, min_dimension_score=4)

        # Build a sample with enough trace content to score well on all dimensions
        sample = Sample(
            id="test_judge_001",
            category=TaskCategory.BUG_FIX,
            language=Language.PYTHON,
            user_input="Fix the bug by updating the isolation level to READ_COMMITTED for consistency",
            assistant_trace=[
                ReasoningStep(
                    goal="Fix isolation level bug",
                    decision="Inspect session configuration first",
                    why="The default may be inherited from the session factory",
                    confidence=0.9,
                ),
                ToolCallStep(
                    tool_name="read_file",
                    call_id="call_0001",
                    arguments={"path": "db/session.py"},
                ),
                ToolResultStep(
                    tool_name="read_file",
                    call_id="call_0001",
                    status=ToolStatus.SUCCESS,
                    summary="Loaded session factory configuration",
                    validation=ValidationCheck(name="read_ok", passed=True),
                    output=ToolOutput(),
                ),
                ToolCallStep(
                    tool_name="edit_file",
                    call_id="call_0004",
                    arguments={
                        "path": "db/session.py",
                        "old_string": '"AUTOCOMMIT"',
                        "new_string": '"READ_COMMITTED"',
                    },
                ),
                ToolResultStep(
                    tool_name="edit_file",
                    call_id="call_0004",
                    status=ToolStatus.SUCCESS,
                    summary="Applied isolation level change",
                    validation=ValidationCheck(name="edit_applied", passed=True),
                    output=ToolOutput(),
                ),
                ToolCallStep(
                    tool_name="test_code",
                    call_id="call_0006",
                    arguments={"target": "tests/test_transactions.py"},
                ),
                ToolResultStep(
                    tool_name="test_code",
                    call_id="call_0006",
                    status=ToolStatus.SUCCESS,
                    exit_code=0,
                    duration_ms=1842,
                    summary="3 tests passed",
                    validation=ValidationCheck(name="exit_code_is_zero", passed=True),
                    output=ToolOutput(stdout="=== 3 passed in 1.84s ==="),
                ),
                FinalStep(
                    content="Updated the default isolation level and verified tests pass",
                    grounded_in=["call_0004", "call_0006"],
                ),
            ],
            final_response="Updated the default isolation level in db/session.py and verified all tests pass",
        )

        verdict, quality = judge.judge(sample)
        assert verdict == JudgeVerdict.ACCEPT

    def test_judge_rejects_low_quality_sample(self):
        judge = Judge(min_avg_score=4.5, min_dimension_score=4)
        sample = Sample(
            id="test_judge_002",
            category=TaskCategory.BUG_FIX,
            language=Language.PYTHON,
            user_input="x",
            assistant_trace=[],
            final_response="",
        )

        verdict, quality = judge.judge(sample)
        assert verdict == JudgeVerdict.REJECT


class TestJSONLExporter:
    def test_export_and_reload(self, tmp_path):
        output = tmp_path / "test.jsonl"
        # Use a very large max_shard_size so no sharding occurs (all writes go to output_path)
        exporter = JSONLExporter(output, max_shard_size=1_000_000)

        sample = Sample(
            id="sample_001",
            category=TaskCategory.BUG_FIX,
            language=Language.PYTHON,
            user_input="Fix the bug",
            assistant_trace=[
                ToolCallStep(
                    tool_name="read_file",
                    call_id="call_0001",
                    arguments={"path": "main.py"},
                ),
            ],
        )
        sample.quality = Quality(
            score=4.5,
            judge_verdict=JudgeVerdict.ACCEPT,
            dimensions=QualityDimension(),
        )

        result = exporter.export_sample(sample)
        exporter.close()

        assert result.lines_written == 1
        samples = load_jsonl(output)
        assert len(samples) == 1
        assert samples[0].id == "sample_001"

    def test_sharding(self, tmp_path):
        output = tmp_path / "sharded.jsonl"
        exporter = JSONLExporter(output, max_shard_size=2)

        for i in range(5):
            sample = Sample(
                id=f"sample_{i:03d}",
                category=TaskCategory.BUG_FIX,
                language=Language.PYTHON,
                user_input=f"Task {i}",
            )
            exporter.export_sample(sample)

        exporter.close()
        assert len(exporter.output_files) > 1
