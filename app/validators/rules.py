"""Validation rules shared across all validators."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.schema import Sample

# ─── Core rules ────────────────────────────────────────────────────────────────

READ_BEFORE_EDIT = "Must read relevant files before editing"
NO_HALLUCINATED_SUCCESS = "Never claim success without verification"
MINIMAL_PATCH = "Prefer editing existing files over rewriting everything"
FINAL_GROUNDED = "Final response must be grounded in tool results"
NO_INVENTED_OUTPUTS = "Never invent file contents, test results, or diffs"
ONLY_AVAILABLE_TOOLS = "Use only the tools provided in available_tools"
DESTRUCTIVE_WITH_CAUTION = "Avoid destructive commands unless explicitly required"

# ─── Default rules block ──────────────────────────────────────────────────────

DEFAULT_RULES = [
    READ_BEFORE_EDIT,
    NO_HALLUCINATED_SUCCESS,
    MINIMAL_PATCH,
    FINAL_GROUNDED,
    NO_INVENTED_OUTPUTS,
    ONLY_AVAILABLE_TOOLS,
    "Avoid redundant tool calls",
    "Do not edit unrelated files",
    "If tests fail, analyze the failure before editing again",
    "Mention uncertainty when verification is incomplete",
]

# ─── Forbidden behaviors ─────────────────────────────────────────────────────

FORBIDDEN_BEHAVIORS = [
    "invent tool outputs",
    "claim task completion without validation",
    "modify unrelated files",
    "hallucinate file contents",
    "skip verification after a fix",
]

SAFETY_POLICIES = [
    "avoid destructive commands unless explicitly required",
    "do not delete files unless task explicitly requires it",
    "confirm destructive operations before proceeding",
]

# ─── Balance targets ─────────────────────────────────────────────────────────

DATASET_BALANCE = {
    "straightforward_success": 0.40,
    "one_failure_then_repair": 0.25,
    "multiple_iterations": 0.15,
    "partial_completion": 0.10,
    "review_heavy": 0.10,
}

DIMENSION_WEIGHTS = {
    "task_understanding": 1.0,
    "reasoning_quality": 1.0,
    "tool_selection_quality": 1.0,
    "tool_call_correctness": 1.0,
    "tool_result_completeness": 1.0,
    "validation_correctness": 1.0,
    "state_transition_consistency": 1.0,
    "minimal_patch_discipline": 1.0,
    "failure_recovery_quality": 1.0,
    "final_response_honesty": 1.0,
}

# ─── Acceptance threshold ─────────────────────────────────────────────────────

DEFAULT_MIN_AVG_SCORE = 4.5
DEFAULT_MIN_DIMENSION_SCORE = 4
DEFAULT_MIN_EXIT_CODE = 0
DEFAULT_MAX_REVIEW_ROUNDS = 5


def acceptance_check(sample: Sample) -> tuple[bool, list[str]]:
    """Check if a sample passes behavioral acceptance criteria.

    Note: does NOT check quality dimensions (those are the Judge's responsibility).
    Only checks behavioral rules like read-before-edit and non-empty final response.
    """
    issues: list[str] = []

    # Must have at least one read-before-edit pattern
    trace = sample.assistant_trace
    has_read = any(
        s.type == "tool_call" and s.tool_name == "read_file"
        for s in trace
    )
    has_edit = any(
        s.type == "tool_call" and s.tool_name == "edit_file"
        for s in trace
    )
    if has_edit and not has_read:
        issues.append("edit performed without any prior read_file call")

    # Final response must not be empty
    if not sample.final_response.strip():
        issues.append("final_response is empty")

    return (len(issues) == 0, issues)
