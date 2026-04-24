"""SWE-bench transformer — converts SWE-bench instances into golden dataset scenario specs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.schema import TaskCategory, Difficulty, Language

logger = logging.getLogger(__name__)


@dataclass
class ScenarioSpec:
    """A normalized scenario spec produced by a transformer."""
    task_id: str
    category: TaskCategory
    repo_files: list[str]
    user_input: str
    expected_capabilities: list[str]
    allowed_tools: list[str]
    difficulty: Difficulty
    language: Language
    instance_id: str
    metadata: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category.value,
            "repo_files": self.repo_files,
            "user_input": self.user_input,
            "expected_capabilities": self.expected_capabilities,
            "allowed_tools": self.allowed_tools,
            "difficulty": self.difficulty.value,
            "language": self.language.value,
            "instance_id": self.instance_id,
            "metadata": self.metadata,
        }


class SWEbenchTransformer:
    """Transforms SWE-bench and SWE-bench Verified instances into scenario specs.

    SWE-bench instances contain: instance_id, repo, version, problem_statement,
    hunk_text, patch, test_patch, wait_for_answer, environment_setup_command,
    gold_repo_dir, instance_image, problem_statement, hints_text, created_at,
   patch_types, repo_directory_name, environment, LIVE, loc, fail_to_pass, pass_to_pass

    This transformer extracts the relevant fields and maps them to the golden dataset schema.
    """

    CATEGORY_KEYWORDS = {
        TaskCategory.BUG_FIX: ["bug", "fix", "error", "crash", "fail", "incorrect"],
        TaskCategory.FEATURE_ADD: ["add", "implement", "support", "new feature"],
        TaskCategory.REFACTOR_SAFE: ["refactor", "restructure", "cleanup", "rename"],
        TaskCategory.TEST_REPAIR: ["test", "assertion", "coverage"],
        TaskCategory.CONFIG_FIX: ["config", "settings", "environment"],
        TaskCategory.DEPENDENCY_FIX: ["dependency", "version", "import", "package"],
    }

    DEFAULT_TOOLS = [
        "read_file",
        "edit_file",
        "test_code",
        "run_command",
    ]

    def __init__(self) -> None:
        self._call_counter = 0

    def transform_instance(self, instance: dict) -> list[ScenarioSpec]:
        """Convert a raw SWE-bench instance into one or more scenario specs.

        A single SWE-bench instance can generate multiple specs at different
        difficulty levels or targeting different capabilities.
        """
        instance_id = instance.get("instance_id", "unknown")
        problem = instance.get("problem_statement", "")

        category = self._infer_category(problem)
        difficulty = self._infer_difficulty(instance)
        repo_files = self._extract_repo_files(instance)

        spec = ScenarioSpec(
            task_id=f"task_{instance_id}",
            category=category,
            repo_files=repo_files,
            user_input=self._build_user_input(instance),
            expected_capabilities=self._build_capabilities(category, instance),
            allowed_tools=self.DEFAULT_TOOLS,
            difficulty=difficulty,
            language=self._detect_language(instance),
            instance_id=instance_id,
            metadata={
                "repo": instance.get("repo"),
                "version": instance.get("version"),
                "fail_to_pass": instance.get("fail_to_pass", []),
                "pass_to_pass": instance.get("pass_to_pass", []),
                "patch_types": instance.get("patch_types", []),
            },
        )

        return [spec]

    def _infer_category(self, problem: str) -> TaskCategory:
        """Infer the task category from the problem statement."""
        text = problem.lower()
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return category
        return TaskCategory.BUG_FIX

    def _infer_difficulty(self, instance: dict) -> Difficulty:
        """Infer difficulty from test complexity and patch size."""
        fail_tests = instance.get("fail_to_pass", [])
        pass_tests = instance.get("pass_to_pass", [])
        total_tests = len(fail_tests) + len(pass_tests)
        patch_types = instance.get("patch_types", [])

        if total_tests > 10 or len(patch_types) > 3:
            return Difficulty.HARD
        if total_tests > 3:
            return Difficulty.MEDIUM
        return Difficulty.EASY

    def _extract_repo_files(self, instance: dict) -> list[str]:
        """Extract the list of relevant files from the instance."""
        files: list[str] = []

        # SWE-bench may have files mentioned in problem statement
        problem = instance.get("problem_statement", "")
        for line in problem.split("\n"):
            if "file" in line.lower() and ("/" in line or ".py" in line):
                import re
                found = re.findall(r'[\w/\.-]+\.py', line)
                files.extend(found)

        if not files:
            # Fallback: construct from repo and environment
            repo = instance.get("repo", "")
            if repo:
                files.append(f"{repo}/__init__.py")

        return files[:10]  # Cap at 10 files

    def _build_user_input(self, instance: dict) -> str:
        """Build a clean user_input from the problem statement."""
        problem = instance.get("problem_statement", "")
        if not problem:
            return "Fix the issue in the repository"
        # Strip HTML-like tags and truncate
        clean = problem.replace("#", "").replace("*", "").strip()
        if len(clean) > 500:
            clean = clean[:497] + "..."
        return clean

    def _build_capabilities(
        self,
        category: TaskCategory,
        instance: dict,
    ) -> list[str]:
        """Build the list of expected capabilities for this spec."""
        capabilities = ["read_before_edit", "minimal_patch", "test_validation"]

        if category == TaskCategory.BUG_FIX:
            capabilities.append("root_cause_analysis")
        if category == TaskCategory.MULTI_FILE_CHANGE:
            capabilities.append("multi_file_editing")
        if len(instance.get("fail_to_pass", [])) > 2:
            capabilities.append("test_iteration")

        return capabilities

    def _detect_language(self, instance: dict) -> Language:
        """Detect the primary language from the repo or environment."""
        repo = instance.get("repo", "").lower()
        env = instance.get("environment", "").lower()

        if "python" in repo or "python" in env:
            return Language.PYTHON
        if "typescript" in repo or "typescript" in env or "ts" in env:
            return Language.TYPESCRIPT
        if "java" in repo or "java" in env:
            return Language.JAVA
        if "javascript" in repo or "javascript" in env:
            return Language.JAVASCRIPT
        if "csharp" in repo or ".NET" in env:
            return Language.CSHARP
        if "react" in repo or "react" in env:
            return Language.REACT

        return Language.PYTHON  # Default
