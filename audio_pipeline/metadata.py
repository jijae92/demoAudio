from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Sequence

from pydub import AudioSegment

from .chunker import ChunkResult, ChunkingConfig
from .tts_engine import TtsEngine

__all__ = ["MetadataBuilder"]


@dataclass
class MetadataBuilder:
    engine: TtsEngine
    config: ChunkingConfig
    output_path: Path

    def build_metadata(
        self,
        *,
        chunks: Sequence[ChunkResult],
        final_segment: AudioSegment,
        final_output: Path,
        options: Dict[str, object],
    ) -> Dict[str, object]:
        total_retries = sum(chunk.retries for chunk in chunks)
        retries_by_chunk = {
            chunk.file_path.name: chunk.retries for chunk in chunks if chunk.retries
        }

        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "engine": self.engine.descriptor(),
            "sample_rate": self.engine.expected_sample_rate,
            "channels": self.engine.expected_channels,
            "sample_width": self.engine.expected_sample_width,
            "format": self.engine.audio_format,
            "target_duration_sec": options.get("target_duration_sec"),
            "max_duration_sec": options.get("max_duration_sec"),
            "silence_gap_ms": options.get("silence_gap_ms"),
            "input_path": str(options.get("input_path")) if options.get("input_path") else None,
            "chunks": [
                {
                    "file": chunk.file_path.name,
                    "ms": chunk.duration_ms,
                    "sentences": chunk.sentence_count,
                    "start_ms": chunk.start_ms,
                    "end_ms": chunk.end_ms,
                    "retries": chunk.retries,
                }
                for chunk in chunks
            ],
            "final_output": str(final_output),
            "final_ms": len(final_segment),
            "retries": {"total": total_retries, "by_chunk": retries_by_chunk},
            "config": {
                "chunk_directory": str(self.config.chunk_directory),
                "keep_chunks": self.config.keep_chunks,
                "sentence_pause_ms": self.config.sentence_pause_ms,
            },
        }

        return metadata

    def write_metadata(self, metadata: Dict[str, object]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
