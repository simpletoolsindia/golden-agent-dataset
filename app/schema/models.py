# Core Pydantic models for the golden dataset schema.
# Every trajectory, step, and artifact is typed here so validators
# and pipeline components share a single source of truth.

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator


# ─── Enums ────────────────────────────────────────────────────────────────────


class StepType(str, Enum):
    REASONING = "reasoning"
    DECISION = "decision"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REVIEW = "review"
    FIX = "fix"
    FINAL = "final"


class TaskCategory(str, Enum):
    BUG_FIX = "bug_fix"
    FEATURE_ADD = "feature_add"
    REFACTOR_SAFE = "refactor_safe"
    TEST_REPAIR = "test_repair"
    CONFIG_FIX = "config_fix"
    DEPENDENCY_FIX = "dependency_fix"
    LINT_FIX = "lint_fix"
    SCHEMA_UPDATE = "schema_update"
    API_INTEGRATION = "api_integration"
    TOOL_USE_ONLY = "tool_use_only"
    DEBUGGING_RCA = "debugging_rca"
    MULTI_FILE_CHANGE = "multi_file_change"
    REGRESSION_PREVENTION = "regression_prevention"
    PARTIAL_FAILURE_RECOVERY = "partial_failure_recovery"
    REVIEW_THEN_FIX = "review_then_fix"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Locale(str, Enum):
    EN = "en"


class Tone(str, Enum):
    PROFESSIONAL = "professional"
    CONCISE = "concise"
    EDUCATED = "educated"


class OutputStyle(str, Enum):
    CONCISE = "concise"
    VERBOSE = "verbose"


class ToolStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class ReviewVerdict(str, Enum):
    APPROVED = "approved"
    NEEDS_FIX = "needs_fix"
    REJECTED = "rejected"


class JudgeVerdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    NEEDS_FIX = "needs_fix"


class SampleSource(str, Enum):
    SYNTHETIC = "synthetic"
    SWEBENCH = "swebench"
    TOOLMIND = "toolmind"
    TOUCAN = "toucan"
    HYBRID = "hybrid"


class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    NODE = "node"
    REACT = "react"


# ─── Tool definitions ────────────────────────────────────────────────────────


class ToolDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Unique tool identifier")
    description: str = Field(description="What the tool does")
    arguments_schema: Optional[dict[str, Any]] = Field(
        default=None,
        description="JSON schema for tool arguments",
    )


class Guardrails(BaseModel):
    forbidden_behaviors: list[str] = Field(
        default_factory=list,
        description="Behaviors the agent must never produce",
    )
    safety_policies: list[str] = Field(
        default_factory=list,
        description="Safety policies to follow",
    )


class OutputContract(BaseModel):
    final_response_must_be_grounded: bool = Field(default=True)
    must_reference_successful_validation: bool = Field(default=True)
    must_acknowledge_incomplete_verification: bool = Field(default=True)


class Localization(BaseModel):
    locale: Locale = Field(default=Locale.EN)
    tone: Tone = Field(default=Tone.PROFESSIONAL)
    style: OutputStyle = Field(default=OutputStyle.CONCISE)


# ─── Step models ──────────────────────────────────────────────────────────────


class ReasoningStep(BaseModel):
    type: StepType = StepType.REASONING
    goal: str = Field(description="Current task goal")
    decision: str = Field(description="Chosen next action")
    why: str = Field(description="Why this action was chosen")
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this decision",
    )
    hypotheses: Optional[list[str]] = Field(
        default=None,
        description="Alternative hypotheses considered",
    )


class DecisionStep(BaseModel):
    type: StepType = StepType.DECISION
    based_on: list[str] = Field(
        description="call_ids this decision is based on",
    )
    decision: str = Field(description="Chosen action or tool")
    why: str = Field(description="Why this option was chosen over alternatives")
    rejected_options: Optional[list[str]] = Field(
        default=None,
        description="Options that were considered and rejected",
    )


class ToolCallStep(BaseModel):
    type: StepType = StepType.TOOL_CALL
    tool_name: str = Field(description="Name of the tool to invoke")
    call_id: str = Field(
        description="Unique identifier for this call, e.g. call_0001",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool",
    )


class ValidationCheck(BaseModel):
    name: str
    passed: bool
    expected: Optional[Any] = None
    actual: Optional[Any] = None


class StateUpdate(BaseModel):
    tests_passing: Optional[int] = None
    tests_failing: Optional[int] = None
    task_verified: Optional[bool] = None
    current_phase: Optional[str] = None
    blocked_by: Optional[str] = None
    files_read: Optional[list[str]] = None
    files_modified: Optional[list[str]] = None


class ToolError(BaseModel):
    category: str = Field(description="Error category, e.g. test_failure")
    message: str = Field(description="Human-readable error message")
    retryable: bool = Field(default=False)


class ToolOutput(BaseModel):
    stdout: str = Field(default="")
    stderr: str = Field(default="")
    content: str = Field(default="")
    diff: str = Field(default="")
    artifacts: list[str] = Field(default_factory=list)


class ToolResultStep(BaseModel):
    type: StepType = StepType.TOOL_RESULT
    tool_name: str = Field(description="Name of the tool that produced this result")
    call_id: str = Field(description="Matches the call_id of the corresponding call")
    status: ToolStatus = Field(description="Overall success/failure/partial")
    exit_code: Optional[int] = Field(default=None)
    duration_ms: Optional[int] = Field(default=None)
    input: Optional[dict[str, Any]] = Field(
        default=None,
        description="Inputs used for this call",
    )
    output: ToolOutput = Field(default_factory=ToolOutput)
    summary: str = Field(
        default="",
        description="Short human-readable summary of the result",
    )
    validation: ValidationCheck = Field(
        description="Outcome of schema and content validation",
    )
    state_update: Optional[StateUpdate] = Field(default=None)
    error: Optional[ToolError] = Field(default=None)


class ReviewIssue(BaseModel):
    severity: str = Field(
        description="Severity: low, medium, high, critical",
    )
    message: str = Field(description="Description of the issue")
    step_ref: Optional[str] = Field(
        default=None,
        description="call_id or step index this issue refers to",
    )


class ReviewStep(BaseModel):
    type: StepType = StepType.REVIEW
    reviewer_role: str = Field(
        default="senior_dataset_reviewer",
        description="Role of the reviewer agent",
    )
    verdict: ReviewVerdict = Field(description="Overall review verdict")
    issues: list[ReviewIssue] = Field(default_factory=list)
    recommended_next_step: Optional[str] = Field(default=None)
    summary: Optional[str] = Field(default=None)


class FixStep(BaseModel):
    type: StepType = StepType.FIX
    fix_strategy: str = Field(description="What the repair agent will do")
    addresses_review_issues: list[str] = Field(
        default_factory=list,
        description="review_issue_* IDs being addressed",
    )


class FinalStep(BaseModel):
    type: StepType = StepType.FINAL
    content: str = Field(description="Final response to the user")
    grounded_in: list[str] = Field(
        default_factory=list,
        description="call_ids that this answer is based on",
    )


# ─── Quality scores ──────────────────────────────────────────────────────────


class QualityDimension(BaseModel):
    task_understanding: int = Field(ge=0, le=5, default=0)
    reasoning_quality: int = Field(ge=0, le=5, default=0)
    tool_selection_quality: int = Field(ge=0, le=5, default=0)
    tool_call_correctness: int = Field(ge=0, le=5, default=0)
    tool_result_completeness: int = Field(ge=0, le=5, default=0)
    validation_correctness: int = Field(ge=0, le=5, default=0)
    state_transition_consistency: int = Field(ge=0, le=5, default=0)
    minimal_patch_discipline: int = Field(ge=0, le=5, default=0)
    failure_recovery_quality: int = Field(ge=0, le=5, default=0)
    final_response_honesty: int = Field(ge=0, le=5, default=0)

    @property
    def average(self) -> float:
        values = [
            self.task_understanding,
            self.reasoning_quality,
            self.tool_selection_quality,
            self.tool_call_correctness,
            self.tool_result_completeness,
            self.validation_correctness,
            self.state_transition_consistency,
            self.minimal_patch_discipline,
            self.failure_recovery_quality,
            self.final_response_honesty,
        ]
        return sum(values) / len(values)

    @property
    def min_score(self) -> int:
        return min(
            self.task_understanding,
            self.reasoning_quality,
            self.tool_selection_quality,
            self.tool_call_correctness,
            self.tool_result_completeness,
            self.validation_correctness,
            self.state_transition_consistency,
            self.minimal_patch_discipline,
            self.failure_recovery_quality,
            self.final_response_honesty,
        )


class Quality(BaseModel):
    score: float = Field(ge=0.0, le=5.0, default=0.0)
    judge_verdict: JudgeVerdict = Field(default=JudgeVerdict.REJECT)
    dimensions: QualityDimension = Field(default_factory=QualityDimension)
    reasoning: Optional[str] = Field(
        default=None,
        description="Why the judge arrived at this verdict",
    )


class SampleMetadata(BaseModel):
    source: SampleSource = Field(default=SampleSource.SYNTHETIC)
    verified: bool = Field(default=False)
    review_rounds: int = Field(default=0)
    generator_version: Optional[str] = Field(default=None)
    seed_task_id: Optional[str] = Field(default=None)
    tags: list[str] = Field(default_factory=list)


# ─── Top-level sample ────────────────────────────────────────────────────────


class Context(BaseModel):
    repo_files: list[str] = Field(
        default_factory=list,
        description="Files relevant to this task",
    )
    repo_path: Optional[str] = Field(default=None)
    additional_context: Optional[dict[str, Any]] = Field(default=None)


class Sample(BaseModel):
    """Top-level golden dataset sample."""

    id: str = Field(description="Unique sample identifier, e.g. sample_000001")
    category: TaskCategory = Field(description="Task category")
    language: Language = Field(description="Primary language")
    difficulty: Difficulty = Field(default=Difficulty.MEDIUM)
    localization: Localization = Field(default_factory=Localization)
    rules: list[str] = Field(
        default_factory=lambda: [
            "Use only available tools",
            "Do not claim success without verification",
            "Prefer editing existing files over rewriting everything",
        ],
        description="Task-specific rules",
    )
    guardrails: Guardrails = Field(default_factory=Guardrails)
    output_contract: OutputContract = Field(default_factory=OutputContract)
    available_tools: list[ToolDefinition] = Field(default_factory=list)
    user_input: str = Field(description="The task prompt")
    context: Context = Field(default_factory=Context)
    assistant_trace: list = Field(
        default_factory=list,
        description="Ordered list of trace steps",
    )
    final_response: str = Field(
        default="",
        description="Final assistant response",
    )
    quality: Quality = Field(default_factory=Quality)
    metadata: SampleMetadata = Field(default_factory=SampleMetadata)

    @model_validator(mode="after")
    def rehydrate_trace_steps(self) -> "Sample":
        """Re-hydrate assistant_trace items from plain dicts after JSON reload."""
        rehydrated: list = []
        for item in self.assistant_trace:
            if isinstance(item, dict):
                t = item.get("type")
                if t == StepType.REASONING.value:
                    rehydrated.append(ReasoningStep(**item))
                elif t == StepType.DECISION.value:
                    rehydrated.append(DecisionStep(**item))
                elif t == StepType.TOOL_CALL.value:
                    rehydrated.append(ToolCallStep(**item))
                elif t == StepType.TOOL_RESULT.value:
                    rehydrated.append(ToolResultStep(**item))
                elif t == StepType.REVIEW.value:
                    rehydrated.append(ReviewStep(**item))
                elif t == StepType.FIX.value:
                    rehydrated.append(FixStep(**item))
                elif t == StepType.FINAL.value:
                    rehydrated.append(FinalStep(**item))
                else:
                    rehydrated.append(item)
            else:
                rehydrated.append(item)
        self.assistant_trace = rehydrated
        return self

    def get_call_ids(self) -> set:
        ids = set()
        for step in self.assistant_trace:
            if hasattr(step, "call_id"):
                ids.add(step.call_id)
        return ids

    def get_tool_results(self) -> list:
        return [s for s in self.assistant_trace if s.type == StepType.TOOL_RESULT]

    def get_final_answer_step(self):
        for step in reversed(self.assistant_trace):
            if step.type == StepType.FINAL:
                return step
        return None
