"""Generator → Reviewer → RCA → Repair → Judge pipeline components."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class GeneratorConfig:
    model: str = "claude-opus-4-7"
    temperature: float = 0.7
    max_tokens: int = 8192
    system_prompt: str = ""
    max_iterations: int = 10


@dataclass
class ReviewerConfig:
    model: str = "claude-opus-4-7"
    temperature: float = 0.3
    max_issues: int = 10


@dataclass
class RCAAgentConfig:
    model: str = "claude-opus-4-7"
    temperature: float = 0.3


@dataclass
class RepairAgentConfig:
    model: str = "claude-opus-4-7"
    temperature: float = 0.5
    max_fix_attempts: int = 3


@dataclass
class JudgeConfig:
    model: str = "claude-opus-4-7"
    temperature: float = 0.0
    min_avg_score: float = 4.5
    min_dimension_score: int = 4


@dataclass
class PipelineConfig:
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    reviewer: ReviewerConfig = field(default_factory=ReviewerConfig)
    rca: RCAAgentConfig = field(default_factory=RCAAgentConfig)
    repair: RepairAgentConfig = field(default_factory=RepairAgentConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    max_review_rounds: int = 3
    seed_prompts_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        data = yaml.safe_load(path.read_text())
        return cls(
            generator=GeneratorConfig(**data.get("generator", {})),
            reviewer=ReviewerConfig(**data.get("reviewer", {})),
            rca=RCAAgentConfig(**data.get("rca", {})),
            repair=RepairAgentConfig(**data.get("repair", {})),
            judge=JudgeConfig(**data.get("judge", {})),
            max_review_rounds=data.get("max_review_rounds", 3),
            seed_prompts_dir=Path(data["seed_prompts_dir"]) if data.get("seed_prompts_dir") else None,
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
        )

    def to_yaml(self, path: Path) -> None:
        data = {
            "generator": self.generator.__dict__,
            "reviewer": self.reviewer.__dict__,
            "rca": self.rca.__dict__,
            "repair": self.repair.__dict__,
            "judge": self.judge.__dict__,
            "max_review_rounds": self.max_review_rounds,
            "seed_prompts_dir": str(self.seed_prompts_dir) if self.seed_prompts_dir else None,
            "output_dir": str(self.output_dir) if self.output_dir else None,
        }
        path.write_text(yaml.dump(data))

    @staticmethod
    def default_config_path() -> Path:
        return Path(__file__).parent.parent.parent / "configs" / "pipeline.yaml"
