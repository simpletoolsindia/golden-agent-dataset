"""Unit tests for the SWE-bench transformer."""

import pytest

from app.schema import TaskCategory, Difficulty, Language
from app.transformers import SWEbenchTransformer, ScenarioSpec


class TestSWEbenchTransformer:
    def test_transform_basic_instance(self):
        transformer = SWEbenchTransformer()
        instance = {
            "instance_id": "django__django-11099",
            "repo": "django/django",
            "version": "3.2",
            "problem_statement": "Bug in query filter causing incorrect results when using related fields. The filter method should handle nested lookups properly.",
            "fail_to_pass": ["tests/queries/test_query.py::QueryTest::test_nested_lookup"],
            "pass_to_pass": ["tests/queries/test_query.py::QueryTest::test_basic_filter"],
            "patch_types": [" Justin Buyer"],
            "environment": "python 3.9",
        }

        specs = transformer.transform_instance(instance)
        assert len(specs) == 1

        spec = specs[0]
        assert spec.task_id == "task_django__django-11099"
        assert spec.instance_id == "django__django-11099"
        assert spec.user_input
        assert len(spec.repo_files) > 0

    def test_infers_bug_fix_category(self):
        transformer = SWEbenchTransformer()
        instance = {
            "instance_id": "pytest__pytest-123",
            "problem_statement": "Bug: crash when running tests with custom markers",
            "fail_to_pass": [],
            "pass_to_pass": [],
        }

        specs = transformer.transform_instance(instance)
        assert specs[0].category == TaskCategory.BUG_FIX

    def test_infers_difficulty_from_test_count(self):
        transformer = SWEbenchTransformer()
        easy = transformer._infer_difficulty({
            "fail_to_pass": ["test_a"],
            "pass_to_pass": ["test_b"],
        })
        hard = transformer._infer_difficulty({
            "fail_to_pass": ["test_a"] * 15,
            "pass_to_pass": ["test_b"] * 15,
            "patch_types": ["a", "b", "c", "d"],
        })

        assert easy == Difficulty.EASY
        assert hard == Difficulty.HARD

    def test_detects_python_language(self):
        transformer = SWEbenchTransformer()
        assert transformer._detect_language({"repo": "django/django"}) == Language.PYTHON
        assert transformer._detect_language({"environment": "python 3.9"}) == Language.PYTHON
        assert transformer._detect_language({"repo": "microsoft/TypeScript"}) == Language.TYPESCRIPT
