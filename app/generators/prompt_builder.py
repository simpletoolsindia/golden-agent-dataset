"""Prompt templates for the generation pipeline."""

from __future__ import annotations

from typing import Optional

from app.schema import Sample, TaskCategory, Language, Difficulty
from app.validators.rules import DEFAULT_RULES


SYSTEM_PROMPT = """You are a high-quality coding agent that produces golden dataset trajectories.
Follow these rules at all times:
- Use only available tools
- Read relevant files before editing
- Prefer minimal safe edits
- Never claim success without verification
- If tests fail, analyze the failure before editing again
- Final response must be grounded in tool results
- Never invent file contents, test results, or diffs

Output every step as a structured JSON object with a "type" field.
Step types: reasoning, decision, tool_call, tool_result, review, fix, final
"""


USER_PROMPT_TEMPLATE = """Generate a structured trajectory for the following task.

Task: {user_input}
Category: {category}
Language: {language}
Difficulty: {difficulty}
Repo files: {repo_files}
Available tools: {available_tools}
Localization: {locale}, tone={tone}, style={style}

Rules:
{rules}

Output each step as a JSON object. Start with a reasoning step explaining your plan.
"""


REVIEWER_PROMPT = """You are a senior dataset reviewer evaluating a coding-agent trajectory.
Review for:
- read_before_edit behavior
- minimal patch discipline
- verification completeness
- honest final response
- consistent state transitions

Output a JSON review step with verdict: approved | needs_fix | rejected
and list issues with severity: low | medium | high | critical
"""


RCA_PROMPT = """You are an RCA agent. Given a reviewer step, explain why each issue occurred.
Focus on:
- root cause of missing validations
- why reasoning was insufficient
- why state tracking failed
- why final response is not grounded

Output a structured RCA analysis.
"""


REPAIR_PROMPT = """You are a repair agent. Given reviewer issues and RCA analysis,
regenerate the faulty steps of the trajectory. Only fix what is broken.
Output fix steps and corrected steps.
"""


JUDGE_PROMPT = """You are a judge evaluating a coding-agent trajectory sample.
Score each dimension from 0-5:
1. task_understanding
2. reasoning_quality
3. tool_selection_quality
4. tool_call_correctness
5. tool_result_completeness
6. validation_correctness
7. state_transition_consistency
8. minimal_patch_discipline
9. failure_recovery_quality
10. final_response_honesty

Accept if: no dimension below 4, average at least 4.5, all structural checks pass.
Output: verdict (accept|reject|needs_fix), score, reasoning.
"""


class PromptBuilder:
    """Builds structured prompts for each pipeline agent."""

    def build_generator_prompt(self, sample: Sample) -> tuple[str, str]:
        tools = ", ".join(t.name for t in sample.available_tools)
        rules = "\n".join(f"- {r}" for r in sample.rules)
        repo_files = ", ".join(sample.context.repo_files)

        user = USER_PROMPT_TEMPLATE.format(
            user_input=sample.user_input,
            category=sample.category.value,
            language=sample.language.value,
            difficulty=sample.difficulty.value,
            repo_files=repo_files or "none",
            available_tools=tools or "none",
            locale=sample.localization.locale.value,
            tone=sample.localization.tone.value,
            style=sample.localization.style.value,
            rules=rules or "\n".join(f"- {r}" for r in DEFAULT_RULES),
        )
        return SYSTEM_PROMPT, user

    def build_reviewer_prompt(self, sample: Sample) -> tuple[str, str]:
        return REVIEWER_PROMPT, f"Sample ID: {sample.id}\nTrace: {sample.model_dump_json(indent=2)}"

    def build_rca_prompt(self, sample: Sample, review: dict) -> tuple[str, str]:
        return RCA_PROMPT, f"Sample: {sample.id}\nReview issues: {review}"

    def build_repair_prompt(
        self,
        sample: Sample,
        review: dict,
        rca: dict,
    ) -> tuple[str, str]:
        return REPAIR_PROMPT, f"Sample: {sample.id}\nReview: {review}\nRCA: {rca}"

    def build_judge_prompt(self, sample: Sample) -> tuple[str, str]:
        return JUDGE_PROMPT, f"Sample: {sample.model_dump_json(indent=2)}"

    def build_transformer_prompt(self, instance: dict, target_category: str) -> str:
        return f"""Convert this SWE-bench instance into golden dataset scenario specs.

Instance: {instance}

Target category: {target_category}
Language: python

Output a JSON array of scenario specs following the schema.
"""
