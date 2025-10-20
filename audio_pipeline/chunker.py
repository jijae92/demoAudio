from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional

from pydub import AudioSegment

from . import split_text
from .tts_engine import TtsEngine

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """
    Configuration describing how audio chunks should be produced.
    """

    target_duration_ms: int
    max_duration_ms: int
    silence_gap_ms: int = 300
    sentence_pause_ms: int = 0
    chunk_directory: Path = Path("output/chunks")
    chunk_prefix: str = "chunk_"
    chunk_extension: str = ".wav"
    keep_chunks: bool = False
    max_retries: int = 5
    initial_retry_delay: float = 0.5
    retry_backoff_factor: float = 2.0

    def ensure_directories(self) -> None:
        self.chunk_directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ChunkResult:
    index: int
    file_path: Path
    duration_ms: int
    sentence_count: int
    sentences: List[str]
    retries: int
    start_ms: int
    end_ms: int


@dataclass
class SentenceWorkItem:
    text: str
    depth: int = 0


class ChunkBuilder:
    """
    Handles sentence synthesis and accumulation into chunk audio files.
    """

    def __init__(self, engine: TtsEngine, config: ChunkingConfig) -> None:
        self.engine = engine
        self.config = config
        self.config.ensure_directories()

    def build_chunks(self, sentences: Iterable[str]) -> List[ChunkResult]:
        queue: Deque[SentenceWorkItem] = deque(
            SentenceWorkItem(text=s.strip()) for s in sentences if s.strip()
        )
        if not queue:
            return []

        chunk_results: List[ChunkResult] = []
        chunk_index = 1
        total_offset_ms = 0
        total_retries_by_chunk: Dict[int, int] = {}

        current_audio = self._empty_segment()
        current_sentences: List[str] = []
        current_duration_ms = 0

        while queue:
            work_item = queue.popleft()
            text = work_item.text
            logger.debug(
                "Processing next sentence. Current chunk duration %d ms with %d sentences.",
                current_duration_ms,
                len(current_sentences),
            )

            try:
                segment, retries = self._synthesize_with_retry(text)
            except Exception as exc:
                logger.error("Failed to synthesize sentence: %s", text)
                raise

            segment_len = len(segment)
            logger.debug(
                "Sentence '%s' produced %d ms (current chunk duration %d ms, sentences=%d).",
                text,
                segment_len,
                current_duration_ms,
                len(current_sentences),
            )
            if segment_len > self.config.max_duration_ms:
                logger.info(
                    "Sentence produced %sms audio which exceeds max %sms. Attempting to re-split.",
                    segment_len,
                    self.config.max_duration_ms,
                )
                resplit_items = self._resplit_sentence(work_item, segment_len)
                if not resplit_items:
                    raise RuntimeError(
                        "Unable to re-split sentence that exceeds maximum duration."
                    )
                queue.extendleft(reversed(resplit_items))
                continue

            # Determine whether adding this segment would exceed chunk targets.
            next_duration = current_duration_ms + segment_len
            target_limit = self.config.target_duration_ms

            logger.debug(
                "Evaluating chunk boundaries: next_duration=%d, target=%d, max=%d, sentences=%d",
                next_duration,
                target_limit,
                self.config.max_duration_ms,
                len(current_sentences),
            )

            if current_sentences and target_limit > 0 and next_duration > target_limit:
                logger.debug(
                    "Flushing chunk %d at %d ms (target limit %d, next sentence length %d).",
                    chunk_index,
                    current_duration_ms,
                    target_limit,
                    segment_len,
                )
                chunk_results.append(
                    self._flush_chunk(
                        chunk_index,
                        current_audio,
                        current_sentences,
                        total_offset_ms,
                        current_duration_ms,
                        total_retries_by_chunk.get(chunk_index, 0),
                    )
                )
                total_offset_ms += current_duration_ms + self.config.silence_gap_ms
                chunk_index += 1
                current_audio = self._empty_segment()
                current_sentences = []
                current_duration_ms = 0

            if current_duration_ms and current_duration_ms + segment_len > self.config.max_duration_ms:
                logger.debug(
                    "Flushing chunk %d due to max duration (current=%d, next=%d, max=%d).",
                    chunk_index,
                    current_duration_ms,
                    segment_len,
                    self.config.max_duration_ms,
                )
                chunk_results.append(
                    self._flush_chunk(
                        chunk_index,
                        current_audio,
                        current_sentences,
                        total_offset_ms,
                        current_duration_ms,
                        total_retries_by_chunk.get(chunk_index, 0),
                    )
                )
                total_offset_ms += current_duration_ms + self.config.silence_gap_ms
                chunk_index += 1
                current_audio = self._empty_segment()
                current_sentences = []
                current_duration_ms = 0

            # Reset retries counter for new chunk index if necessary.
            total_retries_by_chunk.setdefault(chunk_index, 0)
            total_retries_by_chunk[chunk_index] += retries

            current_audio += segment
            if self.config.sentence_pause_ms > 0:
                current_audio += self._silence_segment(self.config.sentence_pause_ms)
            current_sentences.append(text)
            current_duration_ms += len(segment)
            logger.debug(
                "End of iteration state: duration %d ms, sentences=%s",
                current_duration_ms,
                current_sentences,
            )

        if current_sentences:
            chunk_results.append(
                self._flush_chunk(
                    chunk_index,
                    current_audio,
                    current_sentences,
                    total_offset_ms,
                    current_duration_ms,
                    total_retries_by_chunk.get(chunk_index, 0),
                )
            )

        return chunk_results

    def _flush_chunk(
        self,
        index: int,
        audio: AudioSegment,
        sentences: List[str],
        start_ms: int,
        duration_ms: int,
        retries: int,
    ) -> ChunkResult:
        chunk_name = f"{self.config.chunk_prefix}{index:03d}{self.config.chunk_extension}"
        path = self.config.chunk_directory / chunk_name
        logger.info("Exporting chunk %s (%d sentences, %d ms)", chunk_name, len(sentences), duration_ms)
        audio.export(path, format=self.config.chunk_extension.lstrip("."))
        return ChunkResult(
            index=index,
            file_path=path,
            duration_ms=duration_ms,
            sentence_count=len(sentences),
            sentences=list(sentences),
            retries=retries,
            start_ms=start_ms,
            end_ms=start_ms + duration_ms,
        )

    def _empty_segment(self) -> AudioSegment:
        frame_rate = self.engine.expected_sample_rate or 22050
        return AudioSegment.silent(duration=0, frame_rate=frame_rate)

    def _silence_segment(self, duration_ms: int) -> AudioSegment:
        frame_rate = self.engine.expected_sample_rate or 22050
        return AudioSegment.silent(duration=duration_ms, frame_rate=frame_rate)

    def _synthesize_with_retry(self, text: str) -> tuple[AudioSegment, int]:
        delay = self.config.initial_retry_delay
        attempt = 0
        retries = 0
        while True:
            try:
                segment = self.engine.synthesize_to_segment(text)
                return segment, retries
            except Exception as exc:
                attempt += 1
                if attempt >= self.config.max_retries:
                    logger.error("Synthesis permanently failed after %d attempts.", attempt)
                    raise
                retries += 1
                logger.warning(
                    "Synthesis failed (attempt %d/%d). Retrying in %.2fs.",
                    attempt,
                    self.config.max_retries,
                    delay,
                )
                time.sleep(delay)
                delay *= self.config.retry_backoff_factor

    def _resplit_sentence(
        self, work_item: SentenceWorkItem, produced_duration_ms: int
    ) -> List[SentenceWorkItem]:
        text = work_item.text
        if not text or work_item.depth > 5:
            logger.error("Exceeded maximum resplit depth for sentence: %s", text)
            return [
                SentenceWorkItem(s, work_item.depth + 1)
                for s in split_text.hard_split_by_length(text, max_chars=80)
            ]

        ratio = max(1, produced_duration_ms // max(1, self.config.max_duration_ms))
        dynamic_max_chars = max(40, len(text) // (ratio + 1))
        fragments = split_text.split_long_sentence(text, max_chars=dynamic_max_chars)
        if len(fragments) == 1:
            fragments = split_text.hard_split_by_length(text, max_chars=max(40, dynamic_max_chars // 2))

        logger.debug(
            "Resplitting sentence (depth=%d) into %d fragments using max_chars=%d",
            work_item.depth,
            len(fragments),
            dynamic_max_chars,
        )

        return [SentenceWorkItem(fragment, work_item.depth + 1) for fragment in fragments]
