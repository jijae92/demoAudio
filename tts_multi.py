# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import mimetypes
import os
import struct
from google import genai
from google.genai import types


def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"File saved to: {file_name}")


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-pro-preview-tts"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""Expert: 안녕하세요. 오늘은 코덱스 사용방법을 팟캐스트 형식으로 깊고 넓게 풀어보겠습니다. 코딩 초심자부터 실무자까지 듣고 이해할 수 있도록, 개념부터 배포, 보안과 컴플라이언스까지 차근차근 갑니다. [잠깐 멈춤]

Learner: 좋아요. 코덱스가 정확히 뭐죠? 이름만 들었어요. [잠깐 멈춤]

Expert: 핵심 개념 정의부터요. 코덱스는 브라우저에서 코드 편집, 실행, 깃 연동, 배포 편의 기능을 제공하는 개발 워크스페이스입니다. 쉽게 말해, “코드 전용 협업 스튜디오”예요. [잠깐 멈춤]

Learner: 비유로 설명해 주실 수 있나요? [잠깐 멈춤]"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=[
            "audio",
        ],
        speech_config=types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker="Learner",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Zephyr"
                            )
                        ),
                    ),
                    types.SpeakerVoiceConfig(
                        speaker="Expert",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Charon"
                            )
                        ),
                    ),
                ]
            ),
        ),
    )
    audio_chunks: list[bytes] = []
    audio_mime_type: str | None = None
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        for part in chunk.candidates[0].content.parts:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                inline_data = part.inline_data
                audio_mime_type = inline_data.mime_type or audio_mime_type
                audio_data = inline_data.data
                if isinstance(audio_data, str):
                    audio_data = base64.b64decode(audio_data)
                audio_chunks.append(audio_data)
            elif getattr(part, "text", None):
                print(part.text)

    if not audio_chunks:
        print("No audio data was returned by the model.")
        return

    combined_audio = b"".join(audio_chunks)
    mime_type = audio_mime_type or "audio/wav"
    file_extension = mimetypes.guess_extension(mime_type)
    if file_extension is None:
        file_extension = ".wav"
        output_bytes = convert_to_wav(combined_audio, mime_type)
    else:
        output_bytes = combined_audio

    output_name = f"generated_audio{file_extension}"
    save_binary_file(output_name, output_bytes)

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generates a WAV file header for the given audio data and parameters.

    Args:
        audio_data: The raw audio data as a bytes object.
        mime_type: Mime type of the audio data.

    Returns:
        A bytes object representing the WAV file header.
    """
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size  # 36 bytes for header fields before data chunk size

    # http://soundfile.sapp.org/doc/WaveFormat/

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",          # ChunkID
        chunk_size,       # ChunkSize (total file size - 8 bytes)
        b"WAVE",          # Format
        b"fmt ",          # Subchunk1ID
        16,               # Subchunk1Size (16 for PCM)
        1,                # AudioFormat (1 for PCM)
        num_channels,     # NumChannels
        sample_rate,      # SampleRate
        byte_rate,        # ByteRate
        block_align,      # BlockAlign
        bits_per_sample,  # BitsPerSample
        b"data",          # Subchunk2ID
        data_size         # Subchunk2Size (size of audio data)
    )
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """Parses bits per sample and rate from an audio MIME type string.

    Assumes bits per sample is encoded like "L16" and rate as "rate=xxxxx".

    Args:
        mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000").

    Returns:
        A dictionary with "bits_per_sample" and "rate" keys. Values will be
        integers if found, otherwise None.
    """
    bits_per_sample = 16
    rate = 24000

    # Extract rate from parameters
    parts = mime_type.split(";")
    for param in parts: # Skip the main type part
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                # Handle cases like "rate=" with no value or non-integer value
                pass # Keep rate as default
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass # Keep bits_per_sample as default if conversion fails

    return {"bits_per_sample": bits_per_sample, "rate": rate}


if __name__ == "__main__":
    generate()
