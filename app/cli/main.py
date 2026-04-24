#!/usr/bin/env python3
"""CLI entrypoint for the golden dataset pipeline."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from app.schema import Sample
from app.validators import run_all_validators
from app.generators.trajectory_generator import TrajectoryGenerator
from app.reviewers.sample_reviewer import SampleReviewer
from app.repair.repair_agent import RepairAgent
from app.judge.judge import Judge
from app.transformers.swebench_transformer import SWEbenchTransformer
from app.exporters.jsonl_exporter import JSONLExporter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Golden Agent Dataset Pipeline — generate, review, and export
    high-quality coding-agent trajectories.
    """
    pass


# ─── Generate ────────────────────────────────────────────────────────────────


@cli.command("generate")
@click.option("--config", type=click.Path(exists=True), required=True,
              help="Path to pipeline config YAML")
@click.option("--count", type=int, default=1,
              help="Number of samples to generate")
@click.option("--output", type=click.Path(), default="outputs/generated.jsonl",
              help="Output JSONL path")
def generate(config: Path, count: int, output: Path) -> None:
    """Generate candidate trajectories using the Generator → Reviewer → RCA → Repair loop."""
    from app.generators.trajectory_generator import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    generator = TrajectoryGenerator(cfg)

    output.parent.mkdirp(parents=True, exist_ok=True)
    exporter = JSONLExporter(output)

    for i in range(count):
        sample = generator.generate_one()
        result = exporter.export_sample(sample)
        logger.info(f"[{i+1}/{count}] {sample.id} → {result.verdict} (score={sample.quality.score:.2f})")

    exporter.close()
    logger.info(f"Wrote {count} samples to {output}")


# ─── Validate ────────────────────────────────────────────────────────────────


@cli.command("validate")
@click.argument("input", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Show all issues, not just failures")
def validate(input: Path, verbose: bool) -> None:
    """Validate a JSONL file against the golden schema and all quality checks."""
    passed, failed = 0, 0

    for line in input.read_text().splitlines():
        if not line.strip():
            continue
        sample = Sample.model_validate_json(line)
        issues = run_all_validators(sample)

        if not issues:
            passed += 1
            if verbose:
                logger.info(f"PASS {sample.id}")
        else:
            failed += 1
            for issue in issues:
                logger.error(f"FAIL [{sample.id}] {issue}")

    total = passed + failed
    logger.info(f"Validated {total} samples: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


# ─── Review & Repair ─────────────────────────────────────────────────────────


@cli.command("review")
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), default="outputs/reviewed.jsonl")
@click.option("--max-rounds", type=int, default=3)
def review(input: Path, output: Path, max_rounds: int) -> None:
    """Run the Reviewer → RCA → Repair loop on a JSONL file."""
    reviewer = SampleReviewer()
    repair = RepairAgent()

    output.parent.mkdirp(parents=True, exist_ok=True)
    exporter = JSONLExporter(output)

    for line in input.read_text().splitlines():
        if not line.strip():
            continue
        sample = Sample.model_validate_json(line)

        for round_num in range(1, max_rounds + 1):
            review_step = reviewer.review(sample)
            sample.metadata.review_rounds = round_num

            if review_step.verdict == "approved":
                break

            rca = reviewer.explain_rca(sample, review_step)
            fix = repair.repair(sample, review_step, rca)
            sample.assistant_trace.append(fix)

        exporter.export_sample(sample)
        logger.info(f"Reviewed [{round_num} rounds] {sample.id}")

    exporter.close()
    logger.info(f"Reviewed samples written to {output}")


# ─── Transform ───────────────────────────────────────────────────────────────


@cli.command("transform-swebench")
@click.argument("swebench-jsonl", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), default="outputs/swebench_seeds.jsonl")
@click.option("--filter-labels", multiple=True,
              help="Only include instances matching these labels")
def transform_swebench(swebench_jsonl: Path, output: Path, filter_labels: tuple[str, ...]) -> None:
    """Ingest SWE-bench/SWE-bench Verified JSONL and emit scenario specs."""
    transformer = SWEbenchTransformer()

    output.parent.mkdirp(parents=True, exist_ok=True)
    exporter = JSONLExporter(output)

    count = 0
    for line in swebench_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        instance = json.loads(line)

        if filter_labels and instance.get("instance_id", "") not in filter_labels:
            continue

        specs = transformer.transform_instance(instance)
        for spec in specs:
            exporter.export_dict(spec.model_dump())
            count += 1

    exporter.close()
    logger.info(f"Emitted {count} scenario specs to {output}")


# ─── Judge ──────────────────────────────────────────────────────────────────


@cli.command("judge")
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), default="outputs/judged.jsonl")
@click.option("--min-score", type=float, default=4.0,
              help="Minimum average score to accept")
@click.option("--min-dimension", type=int, default=3,
              help="Minimum score any single dimension can have")
def judge(input: Path, output: Path, min_score: float, min_dimension: int) -> None:
    """Score and filter samples with the judge model."""
    judge_model = Judge(min_score=min_score, min_dimension=min_dimension)

    output.parent.mkdirp(parents=True, exist_ok=True)
    exporter = JSONLExporter(output)

    accepted, rejected = 0, 0
    for line in input.read_text().splitlines():
        if not line.strip():
            continue
        sample = Sample.model_validate_json(line)
        verdict, quality = judge_model.judge(sample)

        sample.quality = quality
        sample.metadata.verified = verdict == "accept"

        exporter.export_sample(sample)

        if verdict == "accept":
            accepted += 1
        else:
            rejected += 1

        logger.info(f"{sample.id} → {verdict} (score={quality.score:.2f})")

    exporter.close()
    logger.info(f"Judged {accepted+rejected} samples: {accepted} accepted, {rejected} rejected")


# ─── Export ─────────────────────────────────────────────────────────────────


@cli.command("export")
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), required=True)
@click.option("--max-shard-size", type=int, default=10_000,
              help="Max lines per shard file")
@click.option("--verified-only", is_flag=True,
              help="Only export samples where metadata.verified is True")
def export(input: Path, output: Path, max_shard_size: int, verified_only: bool) -> None:
    """Shard and export a validated JSONL dataset."""
    exporter = JSONLExporter(output, max_shard_size=max_shard_size)
    kept, skipped = 0, 0

    for line in input.read_text().splitlines():
        if not line.strip():
            continue
        sample = Sample.model_validate_json(line)

        if verified_only and not sample.metadata.verified:
            skipped += 1
            continue

        exporter.export_sample(sample)
        kept += 1

    exporter.close()
    logger.info(f"Exported {kept} samples ({skipped} skipped) to {output}")


if __name__ == "__main__":
    cli()
