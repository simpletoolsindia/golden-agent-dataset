from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

from app.schema import Sample

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    path: Path
    lines_written: int
    verdict: str = "unknown"


class JSONLExporter:
    """Exports samples to JSONL with automatic sharding.

    Each line is one JSON-serialized sample. Shards are created when
    max_shard_size lines is reached.
    """

    def __init__(
        self,
        output_path: Path,
        max_shard_size: int = 10_000,
    ) -> None:
        self.output_path = Path(output_path)
        self.max_shard_size = max_shard_size
        self._shard_index = 0
        self._lines_in_current_shard = 0
        self._buffer: list[str] = []
        self._writer: Path | None = None

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._shouldShard():
            self._open_new_shard()

    def _shouldShard(self) -> bool:
        return self.max_shard_size < 1_000_000

    def _shard_name(self, index: int) -> Path:
        stem = self.output_path.stem
        suffix = self.output_path.suffix
        parent = self.output_path.parent
        return parent / f"{stem}_{index:04d}{suffix}"

    def _open_new_shard(self) -> None:
        self._shard_index += 1
        self._writer = self._shard_name(self._shard_index)
        self._lines_in_current_shard = 0

    def export_sample(self, sample: Sample) -> ExportResult:
        line = sample.model_dump_json()
        return self._write_line(line, sample.quality.judge_verdict.value)

    def export_dict(self, d: dict[str, Any]) -> ExportResult:
        line = json.dumps(d, ensure_ascii=False)
        return self._write_line(line, "spec")

    def _write_line(self, line: str, verdict: str) -> ExportResult:
        if self._lines_in_current_shard >= self.max_shard_size:
            self._flush()
            self._open_new_shard()

        with open(self._writer or self.output_path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.write("\n")

        self._lines_in_current_shard += 1

        return ExportResult(
            path=self._writer or self.output_path,
            lines_written=self._lines_in_current_shard,
            verdict=verdict,
        )

    def _flush(self) -> None:
        if self._buffer and self._writer:
            self._writer.write_text("".join(self._buffer))

    def close(self) -> None:
        self._flush()
        logger.info(
            f"Export complete: {self._shard_index} shard(s), "
            f"{self._lines_in_current_shard} lines in final shard"
        )

    @property
    def output_files(self) -> list[Path]:
        if self._shard_index == 0:
            return [self.output_path]
        return [
            self._shard_name(i)
            for i in range(1, self._shard_index + 1)
        ]


def load_jsonl(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    for line in path.read_text().splitlines():
        if line.strip():
            samples.append(Sample.model_validate_json(line))
    return samples


def count_jsonl(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())
