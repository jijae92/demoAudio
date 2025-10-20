"""
Audio synthesis pipeline utilities.

This package exposes the main building blocks used by the CLI entry point:

- Sentence and phrase splitting utilities (`split_text`).
- Engine abstractions and concrete implementations (`tts_engine`).
- Chunk construction logic (`chunker`).
- Audio merging helpers (`merger`).
- Metadata helpers (`metadata`).
"""

from .split_text import (
    hard_split_by_length,
    split_into_sentences,
    split_long_sentence,
)
from .tts_engine import (
    GoogleGenAITtsEngine,
    MockTtsEngine,
    PollyTtsEngine,
    TtsEngine,
)
from .chunker import ChunkBuilder, ChunkResult, ChunkingConfig
from .merger import merge_audio_chunks
from .metadata import MetadataBuilder

__all__ = [
    "split_into_sentences",
    "split_long_sentence",
    "hard_split_by_length",
    "TtsEngine",
    "PollyTtsEngine",
    "GoogleGenAITtsEngine",
    "MockTtsEngine",
    "ChunkingConfig",
    "ChunkBuilder",
    "ChunkResult",
    "merge_audio_chunks",
    "MetadataBuilder",
]
