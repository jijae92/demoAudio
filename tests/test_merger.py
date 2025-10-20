from pathlib import Path

from pydub import AudioSegment

from audio_pipeline.chunker import ChunkResult
from audio_pipeline.merger import merge_audio_chunks


def test_merge_audio_chunks_inserts_silence(tmp_path):
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()

    durations = [1000, 1500, 800]
    chunks = []

    for index, duration in enumerate(durations, start=1):
        segment = AudioSegment.silent(duration=duration, frame_rate=22050)
        file_path = chunk_dir / f"chunk_{index:03d}.wav"
        segment.export(file_path, format="wav")
        chunks.append(
            ChunkResult(
                index=index,
                file_path=file_path,
                duration_ms=len(segment),
                sentence_count=1,
                sentences=[f"sentence-{index}"],
                retries=0,
                start_ms=(index - 1) * 1000,
                end_ms=(index - 1) * 1000 + len(segment),
            )
        )

    output_path = tmp_path / "merged.wav"
    silence_gap = 200

    merged = merge_audio_chunks(chunks, output_path, silence_gap_ms=silence_gap, output_format="wav")

    assert output_path.exists()
    expected_duration = sum(durations) + silence_gap * (len(durations) - 1)
    assert abs(len(merged) - expected_duration) <= 50
