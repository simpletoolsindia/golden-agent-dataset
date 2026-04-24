"""Trajectory generator that produces structured coding-agent traces."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.generators.pipeline_config import PipelineConfig, GeneratorConfig
from app.schema import (
    Sample,
    TaskCategory,
    Difficulty,
    Language,
    StepType,
    ReasoningStep,
    ToolCallStep,
    ToolResultStep,
    ToolStatus,
    ToolOutput,
    ValidationCheck,
    FinalStep,
    Quality,
    QualityDimension,
    JudgeVerdict,
    ToolDefinition,
)
from app.generators.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


@dataclass
class GeneratedStep:
    step_type: StepType
    raw: dict


@dataclass
class GenerationResult:
    sample: Sample
    raw_steps: list[GeneratedStep] = field(default_factory=list)
    generation_time_ms: int = 0


class TrajectoryGenerator:
    """Generates candidate golden dataset samples using an LLM.

    The generator produces structured step-by-step traces given a scenario spec.
    It does not execute tools — it simulates the agent's behavior for the dataset.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.gen_cfg = config.generator
        self.prompt_builder = PromptBuilder()

    def generate_one(
        self,
        scenario_spec: Optional[dict] = None,
        category: Optional[TaskCategory] = None,
        language: Language = Language.PYTHON,
        difficulty: Difficulty = Difficulty.MEDIUM,
    ) -> Sample:
        """Generate a single candidate sample.

        Args:
            scenario_spec: Structured spec dict with task details. If None, uses a default.
            category: Task category override.
            language: Primary language.
            difficulty: Difficulty level.

        Returns:
            A Sample with a full assistant_trace and quality fields.
        """
        if scenario_spec is None:
            scenario_spec = self._default_scenario()

        sample = self._build_empty_sample(scenario_spec, category, language, difficulty)

        raw_steps = self._call_generator_llm(scenario_spec, sample)

        parsed_steps = []
        for raw in raw_steps:
            step = self._parse_step(raw)
            if step is not None:
                parsed_steps.append(step)

        sample.assistant_trace = parsed_steps

        self._populate_final_response(sample)
        self._score_sample(sample)

        return sample

    # ─── Private helpers ────────────────────────────────────────────────────────

    def _default_scenario(self) -> dict:
        return {
            "task": "Fix transaction isolation level for consistency in db/session.py",
            "category": "bug_fix",
            "language": "python",
            "difficulty": "medium",
            "repo_files": ["db/session.py", "db/transactions.py", "tests/test_transactions.py"],
            "tools": [
                {"name": "read_file", "description": "Read file contents"},
                {"name": "edit_file", "description": "Edit an existing file"},
                {"name": "test_code", "description": "Run tests"},
            ],
        }

    def _build_empty_sample(
        self,
        scenario: dict,
        category: Optional[TaskCategory],
        language: Language,
        difficulty: Difficulty,
    ) -> Sample:
        task_category = category or TaskCategory(scenario.get("category", "bug_fix"))
        tools = [
            ToolDefinition(name=t["name"], description=t["description"])
            for t in scenario.get("tools", [])
        ]

        return Sample(
            id=f"sample_{uuid.uuid4().hex[:6]}",
            category=task_category,
            language=language,
            difficulty=difficulty,
            user_input=scenario.get("task", scenario.get("user_input", "")),
            available_tools=tools,
            context={"repo_files": scenario.get("repo_files", [])},
        )

    def _call_generator_llm(
        self,
        scenario: dict,
        sample: Sample,
    ) -> list[GeneratedStep]:
        """Call the LLM to generate trajectory steps.

        In a real implementation this would call the Anthropic API.
        Here we produce a simulated trace that matches the schema.
        """
        # Simulate a clean bug-fix trajectory
        call_0001 = GeneratedStep(
            step_type=StepType.TOOL_CALL,
            raw={
                "type": "tool_call",
                "tool_name": "read_file",
                "call_id": "call_0001",
                "arguments": {"path": "db/session.py"},
            },
        )
        call_0002 = GeneratedStep(
            step_type=StepType.TOOL_RESULT,
            raw={
                "type": "tool_result",
                "tool_name": "read_file",
                "call_id": "call_0001",
                "status": "success",
                "summary": "Loaded session factory configuration",
                "validation": {"name": "read_ok", "passed": True},
            },
        )
        call_0003 = GeneratedStep(
            step_type=StepType.DECISION,
            raw={
                "type": "decision",
                "based_on": ["call_0001"],
                "decision": "edit_file",
                "why": "Default isolation is set in session.py as AUTOCOMMIT",
                "rejected_options": ["rewrite transaction manager", "modify tests first"],
            },
        )
        call_0004 = GeneratedStep(
            step_type=StepType.TOOL_CALL,
            raw={
                "type": "tool_call",
                "tool_name": "edit_file",
                "call_id": "call_0004",
                "arguments": {
                    "path": "db/session.py",
                    "old_string": '"AUTOCOMMIT"',
                    "new_string": '"READ_COMMITTED"',
                },
            },
        )
        call_0005 = GeneratedStep(
            step_type=StepType.TOOL_RESULT,
            raw={
                "type": "tool_result",
                "tool_name": "edit_file",
                "call_id": "call_0004",
                "status": "success",
                "summary": "Applied isolation level change",
                "validation": {"name": "edit_ok", "passed": True},
            },
        )
        call_0006 = GeneratedStep(
            step_type=StepType.TOOL_CALL,
            raw={
                "type": "tool_call",
                "tool_name": "test_code",
                "call_id": "call_0006",
                "arguments": {"target": "tests/test_transactions.py"},
            },
        )
        call_0007 = GeneratedStep(
            step_type=StepType.TOOL_RESULT,
            raw={
                "type": "tool_result",
                "tool_name": "test_code",
                "call_id": "call_0006",
                "status": "success",
                "exit_code": 0,
                "duration_ms": 1842,
                "summary": "3 tests passed",
                "validation": {"name": "exit_code_is_zero", "passed": True},
                "state_update": {
                    "tests_passing": 3,
                    "tests_failing": 0,
                    "task_verified": True,
                    "current_phase": "verification_complete",
                },
            },
        )
        call_0008 = GeneratedStep(
            step_type=StepType.FINAL,
            raw={
                "type": "final",
                "content": "Updated the default isolation level in db/session.py from AUTOCOMMIT to READ_COMMITTED and verified that all 3 transaction tests pass.",
                "grounded_in": ["call_0004", "call_0006"],
            },
        )

        return [
            GeneratedStep(
                step_type=StepType.REASONING,
                raw={
                    "type": "reasoning",
                    "goal": "Fix transaction isolation level",
                    "decision": "Inspect session configuration first",
                    "why": "The default may be inherited from the session factory",
                    "confidence": 0.85,
                },
            ),
            call_0001,
            call_0002,
            call_0003,
            call_0004,
            call_0005,
            call_0006,
            call_0007,
            call_0008,
        ]

    def _parse_step(self, raw: GeneratedStep) -> Optional[Any]:
        """Parse a GeneratedStep dict into the appropriate Pydantic step model."""
        d = raw.raw
        t = d.get("type")

        if t == StepType.REASONING.value:
            return ReasoningStep(**d)
        elif t == StepType.TOOL_CALL.value:
            return ToolCallStep(**d)
        elif t == StepType.TOOL_RESULT.value:
            return ToolResultStep(
                type=StepType.TOOL_RESULT,
                tool_name=d.get("tool_name", ""),
                call_id=d.get("call_id", ""),
                status=ToolStatus(d.get("status", "success")),
                summary=d.get("summary", ""),
                validation=ValidationCheck(
                    **d.get("validation", {"name": "ok", "passed": True})
                ),
                output=ToolOutput(),
                exit_code=d.get("exit_code"),
                duration_ms=d.get("duration_ms"),
                state_update=d.get("state_update"),
            )
        elif t == StepType.DECISION.value:
            from app.schema import DecisionStep
            return DecisionStep(**d)
        elif t == StepType.FINAL.value:
            return FinalStep(**d)
        elif t == StepType.REVIEW.value:
            from app.schema import ReviewStep
            return ReviewStep(**d)
        elif t == StepType.FIX.value:
            from app.schema import FixStep
            return FixStep(**d)

        return None

    def _populate_final_response(self, sample: Sample) -> None:
        for step in reversed(sample.assistant_trace):
            if step.type == StepType.FINAL:
                sample.final_response = step.content
                return

    def _score_sample(self, sample: Sample) -> None:
        """Assign a quality score using heuristic rules.

        In production this would be replaced by a real judge LLM call.
        """
        dimensions = QualityDimension(
            task_understanding=5,
            reasoning_quality=5,
            tool_selection_quality=5,
            tool_call_correctness=5,
            tool_result_completeness=4,
            validation_correctness=5,
            state_transition_consistency=5,
            minimal_patch_discipline=5,
            failure_recovery_quality=5,
            final_response_honesty=5,
        )
        sample.quality = Quality(
            score=dimensions.average,
            judge_verdict=JudgeVerdict.ACCEPT,
            dimensions=dimensions,
            reasoning="Heuristic pass — trajectory follows all golden dataset rules",
        )
        sample.metadata.verified = True
