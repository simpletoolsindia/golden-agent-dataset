# Core Pydantic models for the golden dataset schema.
# Every trajectory, step, and artifact is typed here so validators
# and pipeline components share a single source of truth.

from __future__ import annotations

from typing import Annotated, Any, Literal, NotRequired

from pydantic import BaseModel, Field, ConfigDict


# ─── Enums ────────────────────────────────────────────────────────────────────


class StepType(str, Literal):
    REASONING = "reasoning"
    DECISION = "decision"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REVIEW = "review"
    FIX = "fix"
    FINAL = "final"


class TaskCategory(str, Literal):
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


class Difficulty(str, Literal):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Locale(str, Literal):
    EN = "en"


class Tone(str, Literal):
    PROFESSIONAL = "professional"
    CONCISE = "concise"
    EDUCATED = "educated"


class OutputStyle(str, Literal):
    CONCISE = "concise"
    VERBOSE = "verbose"


class ToolStatus(str, Literal):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class ReviewVerdict(str, Literal):
    APPROVED = "approved"
    NEEDS_FIX = "needs_fix"
    REJECTED = "rejected"


class JudgeVerdict(str, Literal):
    ACCEPT = "accept"
    REJECT = "reject"
    NEEDS_FIX = "needs_fix"


class SampleSource(str, Literal):
    SYNTHETIC = "synthetic"
    SWEBENCH = "swebench"
    TOOLMIND = "toolmind"
    TOUCAN = "toucan"
    HYBRID = "hybrid"


class Language(str, Literal):
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
    arguments_schema: NotRequired[dict[str, Any]] = Field(
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
    type: Literal[StepType.REASONING] = StepType.REASONING
    goal: str = Field(description="Current task goal")
    decision: str = Field(description="Chosen next action")
    why: str = Field(description="Why this action was chosen")
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this decision",
    )
    hypotheses: NotRequired[list[str]] = Field(
        default=None,
        description="Alternative hypotheses considered",
    )


class DecisionStep(BaseModel):
    type: Literal[StepType.DECISION] = StepType.DECISION
    based_on: list[str] = Field(
        description="call_ids this decision is based on",
    )
    decision: str = Field(description="Chosen action or tool")
    why: str = Field(description="Why this option was chosen over alternatives")
    rejected_options: NotRequired[list[str]] = Field(
        default=None,
        description="Options that were considered and rejected",
    )


class ToolCallStep(BaseModel):
    type: Literal[StepType.TOOL_CALL] = StepType.TOOL_CALL
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
    expected: NotRequired[Any] = None
    actual: NotRequired[Any] = None


class StateUpdate(BaseModel):
    tests_passing: NotRequired[int] = None
    tests_failing: NotRequired[int] = None
    task_verified: NotRequired[bool] = None
    current_phase: NotRequired[str] = None
    blocked_by: NotRequired[str] = None
    files_read: NotRequired[list[str]] = None
    files_modified: NotRequired[list[str]] = None


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
    type: Literal[StepType.TOOL_RESULT] = StepType.TOOL_RESULT
    tool_name: str = Field(description="Name of the tool that produced this result")
    call_id: str = Field(description="Matches the call_id of the corresponding call")
    status: ToolStatus = Field(description="Overall success/failure/partial")
    exit_code: NotRequired[int] = Field(default=None)
    duration_ms: NotRequired[int] = Field(default=None)
    input: NotRequired[dict[str, Any]] = Field(
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
    state_update: NotRequired[StateUpdate] = Field(default=None)
    error: NotRequired[ToolError] = Field(default=None)


class ReviewIssue(BaseModel):
    severity: Literal["low", "medium", "high", "critical"]
    message: str = Field(description="Description of the issue")
    step_ref: NotRequired[str] = Field(
        default=None,
        description="call_id or step index this issue refers to",
    )


class ReviewStep(BaseModel):
    type: Literal[StepType.REVIEW] = StepType.REVIEW
    reviewer_role: str = Field(
        default="senior_dataset_reviewer",
        description="Role of the reviewer agent",
    )
    verdict: ReviewVerdict = Field(description="Overall review verdict")
    issues: list[ReviewIssue] = Field(default_factory=list)
    recommended_next_step: NotRequired[str] = Field(default=None)
    summary: NotRequired[str] = Field(default=None)


class FixStep(BaseModel):
    type: Literal[StepType.FIX] = StepType.FIX
    fix_strategy: str = Field(description="What the repair agent will do")
    addresses_review_issues: list[str] = Field(
        default_factory=list,
        description="review_issue_* IDs being addressed",
    )


class FinalStep(BaseModel):
    type: Literal[StepType.FINAL] = StepType.FINAL
    content: str = Field(description="Final response to the user")
    grounded_in: list[str] = Field(
        default_factory=list,
        description="call_ids that this answer is based on",
    )


# ─── Union step type ─────────────────────────────────────────────────────────


TraceStep = Annotated[
    ReasoningStep
    | DecisionStep
    | ToolCallStep
    | ToolResultStep
    | ReviewStep
    | FixStep
    | FinalStep,
    Field(discriminator="type"),
]


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
    judge_verdict: JudgeVerdict = JudgeVerdict.REJECT
    dimensions: QualityDimension = Field(default_factory=QualityDimension)
    reasoning: NotRequired[str] = Field(
        default=None,
        description="Why the judge arrived at this verdict",
    )


class SampleMetadata(BaseModel):
    source: SampleSource = Field(default=SampleSource.SYNTHETIC)
    verified: bool = Field(default=False)
    review_rounds: int = Field(default=0)
    generator_version: NotRequired[str] = Field(default=None)
    seed_task_id: NotRequired[str] = Field(default=None)
    tags: list[str] = Field(default_factory=list)


# ─── Top-level sample ────────────────────────────────────────────────────────


class Context(BaseModel):
    repo_files: list[str] = Field(
        default_factory=list,
        description="Files relevant to this task",
    )
    repo_path: NotRequired[str] = Field(default=None)
    additional_context: NotRequired[dict[str, Any]] = Field(default=None)


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
    assistant_trace: list[TraceStep] = Field(
        default_factory=list,
        description="Ordered list of trace steps",
    )
    final_response: str = Field(
        default="",
        description="Final assistant response",
    )
    quality: Quality = Field(default_factory=Quality)
    metadata: SampleMetadata = Field(default_factory=SampleMetadata)

    def get_call_ids(self) -> set[str]:
        """Return all call_ids present in the trace."""
        ids: set[str] = set()
        for step in self.assistant_trace:
            if hasattr(step, "call_id"):
                ids.add(step.call_id)
        return ids

    def get_tool_results(self) -> list[ToolResultStep]:
        return [s for s in self.assistant_trace if s.type == StepType.TOOL_RESULT]

    def get_final_answer_step(self) -> FinalStep | None:
        for step in reversed(self.assistant_trace):
            if step.type == StepType.FINAL:
                return step
        return None
