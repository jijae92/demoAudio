from __future__ import annotations

import base64
import io
import logging
import mimetypes
from abc import ABC, abstractmethod
from typing import Dict, Optional

from pydub import AudioSegment

logger = logging.getLogger(__name__)

__all__ = [
    "TtsEngine",
    "PollyTtsEngine",
    "GoogleGenAITtsEngine",
    "MockTtsEngine",
]


class TtsEngine(ABC):
    """
    Thin abstraction over a text-to-speech engine that returns AudioSegment objects.
    """

    def __init__(
        self,
        *,
        expected_sample_rate: Optional[int] = None,
        expected_channels: Optional[int] = None,
        expected_sample_width: Optional[int] = None,
        audio_format: str = "wav",
    ) -> None:
        self.expected_sample_rate = expected_sample_rate
        self.expected_channels = expected_channels
        self.expected_sample_width = expected_sample_width
        self.audio_format = audio_format

    @abstractmethod
    def synthesize_to_segment(self, text: str, is_ssml: bool = False) -> AudioSegment:
        """
        Convert text (or SSML when supported) into a pydub AudioSegment.
        """

    def descriptor(self) -> str:
        return self.__class__.__name__

    def _validate_segment(self, segment: AudioSegment) -> AudioSegment:
        """
        Ensures the returning AudioSegment sticks to the expected format.

        The first call establishes the reference format unless the engine was initialised
        with explicit expectations. Subsequent calls must match.
        """
        if self.expected_sample_rate is None:
            self.expected_sample_rate = segment.frame_rate
        elif segment.frame_rate != self.expected_sample_rate:
            raise ValueError(
                f"Engine {self.descriptor()} returned frame rate {segment.frame_rate}, "
                f"expected {self.expected_sample_rate}"
            )

        if self.expected_channels is None:
            self.expected_channels = segment.channels
        elif segment.channels != self.expected_channels:
            raise ValueError(
                f"Engine {self.descriptor()} returned channels {segment.channels}, "
                f"expected {self.expected_channels}"
            )

        if self.expected_sample_width is None:
            self.expected_sample_width = segment.sample_width
        elif segment.sample_width != self.expected_sample_width:
            raise ValueError(
                f"Engine {self.descriptor()} returned sample width {segment.sample_width}, "
                f"expected {self.expected_sample_width}"
            )

        return segment


class MockTtsEngine(TtsEngine):
    """
    Lightweight mock for tests. Generates silent segments of predictable lengths.
    """

    def __init__(
        self,
        durations_ms: Optional[Dict[str, int]] = None,
        *,
        base_duration_ms: int = 500,
        per_char_ms: int = 30,
        sample_rate: int = 22050,
        channels: int = 1,
        sample_width: int = 2,
    ) -> None:
        super().__init__(
            expected_sample_rate=sample_rate,
            expected_channels=channels,
            expected_sample_width=sample_width,
            audio_format="wav",
        )
        self._durations_ms = durations_ms or {}
        self._base_duration_ms = base_duration_ms
        self._per_char_ms = per_char_ms

    def synthesize_to_segment(self, text: str, is_ssml: bool = False) -> AudioSegment:
        duration = self._durations_ms.get(
            text, self._base_duration_ms + max(0, len(text)) * self._per_char_ms
        )
        segment = AudioSegment.silent(duration=duration, frame_rate=self.expected_sample_rate)  # type: ignore[arg-type]
        return self._validate_segment(segment)


class PollyTtsEngine(TtsEngine):
    """
    Amazon Polly implementation that produces PCM/WAV AudioSegment instances.
    """

    def __init__(
        self,
        *,
        voice_id: str,
        engine: str = "neural",
        language_code: Optional[str] = None,
        sample_rate: int = 22050,
        output_format: str = "pcm",
        boto3_client: Optional[object] = None,
        text_type: str = "text",
        channels: int = 1,
    ) -> None:
        super().__init__(
            expected_sample_rate=sample_rate,
            expected_channels=channels,
            expected_sample_width=2,
            audio_format="wav" if output_format.lower() == "pcm" else output_format,
        )
        try:
            import boto3  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "boto3 is required for PollyTtsEngine but is not installed."
            ) from exc

        self._client = boto3_client or boto3.client("polly")
        self._voice_id = voice_id
        self._engine = engine
        self._language_code = language_code
        self._sample_rate = sample_rate
        self._output_format = output_format
        self._text_type = text_type

    def synthesize_to_segment(self, text: str, is_ssml: bool = False) -> AudioSegment:
        params = {
            "Engine": self._engine,
            "VoiceId": self._voice_id,
            "OutputFormat": self._output_format,
            "SampleRate": str(self._sample_rate),
            "Text": text,
            "TextType": "ssml" if is_ssml else self._text_type,
        }
        if self._language_code:
            params["LanguageCode"] = self._language_code

        logger.debug("Polly request params: %s", {k: v for k, v in params.items() if k != "Text"})
        response = self._client.synthesize_speech(**params)  # type: ignore[arg-type]
        stream = response.get("AudioStream")
        if stream is None:
            raise RuntimeError("Polly response did not include AudioStream.")

        audio_bytes = stream.read() if hasattr(stream, "read") else stream
        if not audio_bytes:
            raise RuntimeError("Polly returned empty audio stream.")

        format_lower = self._output_format.lower()
        if format_lower == "pcm":
            segment = AudioSegment(
                data=audio_bytes,
                sample_width=self.expected_sample_width or 2,
                frame_rate=self.expected_sample_rate or self._sample_rate,
                channels=self.expected_channels or 1,
            )
        else:
            segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format_lower)

        return self._validate_segment(segment)


class GoogleGenAITtsEngine(TtsEngine):
    """
    Google Generative AI TTS implementation using the ``google-genai`` client.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-2.5-pro-preview-tts",
        voice_name: Optional[str] = None,
        sample_rate: int = 24000,
        audio_mime_type: str = "audio/wav",
        language_code: Optional[str] = None,
    ) -> None:
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "google-genai is required for GoogleGenAITtsEngine but is not installed."
            ) from exc

        super().__init__(
            expected_sample_rate=sample_rate,
            expected_channels=1,
            expected_sample_width=2,
            audio_format="wav",
        )
        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._model = model
        self._voice_name = voice_name
        self._language_code = language_code
        self._mime_type = audio_mime_type

    def synthesize_to_segment(self, text: str, is_ssml: bool = False) -> AudioSegment:
        types = self._types
        part = types.Part.from_text(text=text)
        content = types.Content(role="user", parts=[part])
        voice_config = None
        if self._voice_name:
            voice_config = types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker="narrator",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=self._voice_name
                            )
                        ),
                    )
                ]
            )
        speech_config = types.SpeechConfig(
            language_code=self._language_code,
            multi_speaker_voice_config=voice_config,
        )
        generate_config = types.GenerateContentConfig(
            response_modalities=["audio"],
            speech_config=speech_config,
        )

        audio_chunks = []
        mime_type: Optional[str] = None
        for chunk in self._client.models.generate_content_stream(
            model=self._model,
            contents=[content],
            config=generate_config,
        ):
            candidate = (chunk.candidates or [None])[0]
            if not candidate or not candidate.content or not candidate.content.parts:
                continue
            for response_part in candidate.content.parts:
                inline = getattr(response_part, "inline_data", None)
                if inline and inline.data:
                    mime_type = inline.mime_type or mime_type
                    data = inline.data
                    if isinstance(data, str):
                        data = base64.b64decode(data)
                    audio_chunks.append(data)

        if not audio_chunks:
            raise RuntimeError("Google GenAI returned no audio data.")

        audio_bytes = b"".join(audio_chunks)
        mime = mime_type or self._mime_type
        segment = _audio_bytes_to_segment(audio_bytes, mime, default_rate=self.expected_sample_rate or 24000)
        return self._validate_segment(segment)


def _audio_bytes_to_segment(data: bytes, mime_type: str, *, default_rate: int) -> AudioSegment:
    mime_type = mime_type or "audio/wav"
    if mime_type.startswith("audio/L"):
        params = _parse_linear_pcm_mime(mime_type)
        frame_rate = params.get("rate", default_rate)
        sample_width = params.get("sample_width", 2)
        channels = params.get("channels", 1)
        return AudioSegment(
            data=data,
            sample_width=sample_width,
            frame_rate=frame_rate,
            channels=channels,
        )

    guessed = (mimetypes.guess_extension(mime_type) or "").lstrip(".")
    fmt = guessed or mime_type.split("/")[-1]
    return AudioSegment.from_file(io.BytesIO(data), format=fmt)


def _parse_linear_pcm_mime(mime_type: str) -> Dict[str, int]:
    params: Dict[str, int] = {"rate": 24000, "sample_width": 2, "channels": 1}
    fragments = [fragment.strip() for fragment in mime_type.split(";")]
    for fragment in fragments:
        if fragment.lower().startswith("rate="):
            try:
                params["rate"] = int(fragment.split("=", 1)[1])
            except ValueError:
                logger.warning("Unable to parse rate from mime type %s", mime_type)
        elif fragment.lower().startswith("channels="):
            try:
                params["channels"] = int(fragment.split("=", 1)[1])
            except ValueError:
                logger.warning("Unable to parse channels from mime type %s", mime_type)
        elif fragment.lower().startswith("audio/l"):
            try:
                bits = int(fragment.split("L", 1)[1])
                params["sample_width"] = max(1, bits // 8)
            except ValueError:
                logger.warning("Unable to parse bits from mime type %s", mime_type)
    return params
