from pathlib import Path

from audio_pipeline.chunker import ChunkBuilder, ChunkingConfig
from audio_pipeline.tts_engine import MockTtsEngine


def test_chunk_builder_respects_target_duration(tmp_path):
    sentences = [
        "첫 번째 문장입니다",
        "두 번째 문장은 조금 더 깁니다",
        "세 번째 문장도 추가합니다",
    ]
    durations = {
        sentences[0]: 60000,
        sentences[1]: 50000,
        sentences[2]: 50000,
    }

    engine = MockTtsEngine(durations_ms=durations, sample_rate=22050)
    config = ChunkingConfig(
        target_duration_ms=120000,
        max_duration_ms=180000,
        silence_gap_ms=250,
        sentence_pause_ms=0,
        chunk_directory=tmp_path / "chunks",
    )

    builder = ChunkBuilder(engine, config)
    chunks = builder.build_chunks(sentences)

    # Should produce two chunks: first two sentences together, third on its own.
    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.duration_ms <= config.max_duration_ms
        assert chunk.file_path.exists()

    # First chunk should stay within the target duration of 120s.
    assert chunks[0].duration_ms <= config.target_duration_ms
