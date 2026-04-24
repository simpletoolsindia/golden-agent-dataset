# Golden Agent Dataset

A production pipeline for generating high-quality coding-agent golden datasets.

## What this is

This project builds a structured dataset for training a coding agent that can:
- Call tools deterministically and with correct arguments
- Follow read-before-edit behavior
- Apply minimal, safe code changes
- Run and interpret test results
- Debug, find root causes, and iterate through fix loops
- Maintain structured state across steps
- Ground final answers in actual tool results

## Architecture

```
golden-agent-dataset/
├── app/
│   ├── schema/          # Pydantic models (Sample, TraceStep subtypes, Quality)
│   ├── validators/      # Structural, behavioral, validation validators
│   ├── generators/      # TrajectoryGenerator, PipelineConfig, PromptBuilder
│   ├── reviewers/       # SampleReviewer with heuristic issue detection
│   ├── repair/          # RepairAgent with heuristic fix application
│   ├── judge/           # Judge scoring on 10 dimensions
│   ├── transformers/    # SWE-bench → scenario spec transformer
│   ├── exporters/       # JSONL exporter with sharding
│   └── cli/            # CLI entrypoints
├── configs/            # Pipeline, tools, and balance configs
├── prompts/            # System prompts for each pipeline agent
├── examples/           # Task templates and sample trajectories
├── tests/              # Unit tests per module
└── outputs/            # Generated dataset output
```

## Pipeline

```
Scenario Spec
  → TrajectoryGenerator (produces raw structured trace)
  → SampleReviewer (finds quality issues)
  → RCA Analysis (explains root causes)
  → RepairAgent (regenerates faulty steps)
  → Judge (scores on 10 dimensions, accepts/rejects)
  → JSONLExporter (shards and exports)
```

## Quick start

```bash
# Install
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Generate sample trajectories
python -m app.cli.main generate --config configs/pipeline.yaml --count 10

# Validate generated output
python -m app.cli.main validate outputs/generated.jsonl

# Judge and filter
python -m app.cli.main judge outputs/generated.jsonl --output outputs/judged.jsonl

# Transform SWE-bench seeds
python -m app.cli.main transform-swebench /path/to/swebench.jsonl --output outputs/swebench_seeds.jsonl
```

## CLI commands

| Command | Description |
|---|---|
| `generate` | Generate candidate trajectories |
| `validate` | Validate JSONL against schema and rules |
| `review` | Run Reviewer → RCA → Repair loop |
| `transform-swebench` | Ingest SWE-bench JSONL into scenario specs |
| `judge` | Score and filter samples by quality |
| `export` | Shard and export validated JSONL |

## Schema

Every sample in the dataset is a `Sample` with:
- `id`, `category`, `language`, `difficulty`, `localization`
- `rules`, `guardrails`, `output_contract`, `available_tools`
- `user_input` and `context`
- `assistant_trace`: ordered list of typed steps
  - `reasoning`, `decision`, `tool_call`, `tool_result`, `review`, `fix`, `final`
- `final_response`, `quality`, `metadata`

The `tool_result` step is the most important — it stores structured outcome data:
```python
ToolResultStep(
    tool_name="test_code",
    call_id="call_0003",
    status="success",
    exit_code=0,
    duration_ms=1842,
    output={stdout, stderr, content, diff, artifacts},
    summary="3 tests passed",
    validation={name, passed, expected, actual},
    state_update={tests_passing, tests_failing, current_phase, ...},
    error=None,
)
```

## Quality rubric

10 dimensions scored 0–5:
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

Acceptance: no dimension below 4, average ≥ 4.5, all structural checks pass.

## Current status

**Phase 1 complete** — schema, validators, generator, reviewer, RCA, repair, judge, SWE-bench transformer, JSONL exporter, CLI.

**33 tests passing.** Pipeline generates 10/10 valid samples that pass all structural and behavioral validators.

```bash
# Verify
python -m pytest tests/ -v

# Generate and validate 10 samples
python -c "
from pathlib import Path
from app.schema import Sample
from app.validators import run_all_validators
from app.generators import TrajectoryGenerator, PipelineConfig
from app.judge import Judge

gen = TrajectoryGenerator(PipelineConfig())
judge = Judge(min_avg_score=4.5, min_dimension_score=4)
out = Path('outputs/generated.jsonl')
out.write_text('')
for i in range(10):
    s = gen.generate_one()
    v, q = judge.judge(s)
    s.quality, s.metadata.verified = q, v.value=='accept'
    out.open('a').write(s.model_dump_json()+'\n')
lines = [l for l in out.read_text().splitlines() if l]
ok = sum(1 for l in lines if not run_all_validators(Sample.model_validate_json(l)))
print(f'{ok}/{len(lines)} samples pass all validators')
"
```

## Repository

https://github.com/simpletoolsindia/golden-agent-dataset
