from traveler_agent.emoji_analysis import emoji_profile, emoji_marker_count, has_emoticon


def test_emoji_positive_negative_markers_without_sentiment_inference():
    result = emoji_profile(["Un paseo excelente 😍", "Qué espera 😡", "Sin símbolos"])
    assert result["rows"] == 3
    assert result["rows_with_emoji_markers"] == 2
    assert result["emoji_marker_codepoints"] == 2
    assert "sentiment" not in result


def test_ascii_emoticon_distinguished_from_unicode():
    p = emoji_profile(["Muy bueno :)", "Solo texto", "No estuvo mal :("])
    assert p["rows_with_ascii_emoticons"] == 2
    assert p["rows_with_emoji_markers"] == 0


def test_negation_and_irony_not_automatically_labeled():
    result = emoji_profile(["Excelente... dos horas esperando 🙃", "No fue malo 😂"])
    assert result["rows_with_emoji_markers"] == 2
    assert "predicted_sentiment" not in result


def test_numeric_labels_remain_raw_without_semantic_mapping():
    result = emoji_profile(["Bien 😍", "Mal", "Regular 🙃"], labels=[2, 0, 1])
    assert [x["label_raw"] for x in result["raw_label_groups"]] == ["0", "1", "2"]
    assert result["raw_label_groups"][1]["rows_with_emoji_markers"] == 1


def test_invalid_length_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="row counts"):
        emoji_profile(["A", "B"], labels=[1])


def test_emoji_heuristic_limitations_explicit():
    assert emoji_marker_count("👍🏽") >= 1
    assert has_emoticon("Hola :)")
    assert "unicode_block" in emoji_profile([])["method"]
