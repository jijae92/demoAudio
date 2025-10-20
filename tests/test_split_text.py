from audio_pipeline.split_text import split_into_sentences


def test_split_into_sentences_handles_long_phrases():
    text = (
        "첫 번째 문장입니다. "
        "두 번째 문장은 꽤 길어서 쉼표와 여러 구절을 포함하고, "
        "자동으로 분할되어야 합니다. "
        "세 번째 문장도 있습니다!"
    )

    sentences = split_into_sentences(text, max_chars=40)

    # Expect more than the original 3 sentences because the long middle sentence is split.
    assert len(sentences) >= 4
    assert all(sentence.strip() for sentence in sentences)
    assert any("쉼표" in sentence for sentence in sentences)
