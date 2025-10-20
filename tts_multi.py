#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Sequence

from audio_pipeline.chunker import ChunkBuilder, ChunkingConfig, ChunkResult
from audio_pipeline.metadata import MetadataBuilder
from audio_pipeline.merger import merge_audio_chunks
from audio_pipeline.split_text import split_into_sentences
from audio_pipeline.tts_engine import (
    GoogleGenAITtsEngine,
    MockTtsEngine,
    PollyTtsEngine,
    TtsEngine,
)

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-chunk TTS synthesis pipeline.")
    parser.add_argument("--input", required=True, help="Input text or SSML file path.")
    parser.add_argument("--input-encoding", default="utf-8", help="Encoding used for input file.")
    parser.add_argument("--max-duration-sec", type=int, default=300, help="Maximum duration per chunk in seconds.")
    parser.add_argument("--target-duration-sec", type=int, default=180, help="Target duration per chunk in seconds.")
    parser.add_argument("--silence-gap-ms", type=int, default=300, help="Silence inserted between chunks in milliseconds.")
    parser.add_argument("--sentence-pause-ms", type=int, default=0, help="Optional pause inserted between sentences within a chunk.")
    parser.add_argument("--merge-output", default="./output/final_merged.wav", help="Path for merged output audio.")
    parser.add_argument("--metadata-output", default="./output/metadata.json", help="Path for metadata JSON output.")
    parser.add_argument("--chunk-dir", default="./output/chunks", help="Directory to store intermediate chunk files.")
    parser.add_argument("--keep-chunks", action="store_true", help="Keep chunk files after merging.")
    parser.add_argument("--engine", default="google_genai", help="TTS engine to use (google_genai, polly, mock).")
    parser.add_argument("--api-key", help="API key for engines that require one (e.g. Google GenAI).")
    parser.add_argument("--google-model", default="gemini-2.5-pro-preview-tts", help="Google GenAI model name.")
    parser.add_argument("--voice-id", help="Voice identifier (engine specific).")
    parser.add_argument("--language-code", help="Language code hint for engine.")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Expected sample rate for audio segments.")
    parser.add_argument("--format", default="pcm", help="Underlying engine format (pcm/mp3/ogg_vorbis).")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum synthesis retries per sentence.")
    parser.add_argument("--retry-initial-delay", type=float, default=0.5, help="Initial retry delay in seconds.")
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="Multiplier for retry backoff.")
    parser.add_argument("--concurrency", type=int, default=1, help="Reserved for future parallel synthesis control.")
    parser.add_argument("--is-ssml", action="store_true", help="Treat input as SSML rather than plain text.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    return parser.parse_args(argv)


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def load_input_text(path: Path, encoding: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    return path.read_text(encoding=encoding)


def create_engine(args: argparse.Namespace) -> TtsEngine:
    engine_name = (args.engine or "").lower()
    sample_rate = args.sample_rate
    if engine_name in {"mock", "dummy"}:
        return MockTtsEngine(sample_rate=sample_rate)

    if engine_name in {"polly", "aws_polly"}:
        if not args.voice_id:
            raise ValueError("--voice-id is required when using the Polly engine.")
        return PollyTtsEngine(
            voice_id=args.voice_id,
            engine="neural",
            language_code=args.language_code,
            sample_rate=sample_rate,
            output_format=args.format,
            channels=1,
        )

    if engine_name in {"google", "google_genai", "gemini"}:
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_GENAI_API_KEY")
        if not api_key:
            raise ValueError("Google GenAI engine requires an API key (use --api-key or GEMINI_API_KEY env var).")
        return GoogleGenAITtsEngine(
            api_key=api_key,
            model=args.google_model,
            voice_name=args.voice_id,
            sample_rate=sample_rate,
            audio_mime_type="audio/wav" if args.format == "pcm" else f"audio/{args.format}",
            language_code=args.language_code,
        )

    raise ValueError(f"Unsupported engine: {args.engine}")


def build_metadata_options(args: argparse.Namespace, input_path: Path) -> dict:
    return {
        "input_path": input_path,
        "target_duration_sec": args.target_duration_sec,
        "max_duration_sec": args.max_duration_sec,
        "silence_gap_ms": args.silence_gap_ms,
        "engine": args.engine,
        "voice_id": args.voice_id,
        "language_code": args.language_code,
        "format": args.format,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    configure_logging(args.debug)

    if args.max_duration_sec <= 0:
        raise ValueError("--max-duration-sec must be positive.")
    if args.target_duration_sec <= 0 or args.target_duration_sec > args.max_duration_sec:
        logger.warning(
            "Adjusting target duration to be within (0, max]. "
            "Received target=%s, max=%s.",
            args.target_duration_sec,
            args.max_duration_sec,
        )
        args.target_duration_sec = min(args.max_duration_sec, max(1, args.target_duration_sec))

    input_path = Path(args.input)
    text = load_input_text(input_path, args.input_encoding)
    is_ssml = args.is_ssml

    if is_ssml:
        sentences = [text]
    else:
        sentences = split_into_sentences(text)
    if not sentences:
        logger.warning("No sentences found in input. Nothing to synthesize.")
        return 0

    engine = create_engine(args)
    chunk_config = ChunkingConfig(
        target_duration_ms=args.target_duration_sec * 1000,
        max_duration_ms=args.max_duration_sec * 1000,
        silence_gap_ms=args.silence_gap_ms,
        sentence_pause_ms=args.sentence_pause_ms,
        chunk_directory=Path(args.chunk_dir),
        keep_chunks=args.keep_chunks,
        max_retries=args.max_retries,
        initial_retry_delay=args.retry_initial_delay,
        retry_backoff_factor=args.retry_backoff,
    )

    builder = ChunkBuilder(engine, chunk_config)
    logger.info("Splitting text into %d sentences.", len(sentences))
    chunk_results = builder.build_chunks(sentences)
    if not chunk_results:
        logger.warning("No audio chunks were produced.")
        return 0

    merge_output_path = Path(args.merge_output)
    output_format = merge_output_path.suffix.lstrip(".").lower() or "wav"

    merged_segment = merge_audio_chunks(
        chunk_results,
        merge_output_path,
        silence_gap_ms=args.silence_gap_ms,
        output_format=output_format,
    )

    metadata_builder = MetadataBuilder(
        engine=engine,
        config=chunk_config,
        output_path=Path(args.metadata_output),
    )
    metadata = metadata_builder.build_metadata(
        chunks=chunk_results,
        final_segment=merged_segment,
        final_output=merge_output_path,
        options=build_metadata_options(args, input_path),
    )
    metadata_builder.write_metadata(metadata)
    logger.info("Metadata written to %s", metadata_builder.output_path)

    if not args.keep_chunks:
        _cleanup_chunks(chunk_results)

    logger.info("Synthesis complete. Final audio saved to %s", merge_output_path)
    return 0


def _cleanup_chunks(chunks: Sequence[ChunkResult]) -> None:
    for chunk in chunks:
        try:
            chunk.file_path.unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to delete chunk %s: %s", chunk.file_path, exc)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)
