from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

from pydub import AudioSegment

from .chunker import ChunkResult

logger = logging.getLogger(__name__)

__all__ = ["merge_audio_chunks"]


def merge_audio_chunks(
    chunks: Sequence[ChunkResult],
    output_path: Path,
    *,
    silence_gap_ms: int = 300,
    output_format: str = "wav",
) -> AudioSegment:
    if not chunks:
        raise ValueError("No chunks provided for merging.")

    merged: AudioSegment | None = None
    chunk_count = len(chunks)

    for idx, chunk in enumerate(chunks):
        segment = AudioSegment.from_file(chunk.file_path)
        merged = segment if merged is None else merged + segment

        if idx < chunk_count - 1 and silence_gap_ms > 0:
            merged += _matching_silence(segment, silence_gap_ms)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.export(output_path, format=output_format)
    logger.info("Merged %d chunks into %s", chunk_count, output_path)
    return merged


def _matching_silence(segment: AudioSegment, duration_ms: int) -> AudioSegment:
    silence = AudioSegment.silent(duration=duration_ms, frame_rate=segment.frame_rate)
    silence = silence.set_channels(segment.channels)
    silence = silence.set_sample_width(segment.sample_width)
    return silence
