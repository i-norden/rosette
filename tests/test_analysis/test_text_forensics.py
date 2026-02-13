from __future__ import annotations

from unittest.mock import patch

from snoopy.analysis.text_forensics import (
    TorturedPhraseMatch,
    TorturedPhraseResult,
    detect_tortured_phrases,
)

# Small phrase dict for tests (the full data file is gitignored)
_TEST_PHRASES = {
    "fake neural system": "artificial neural network",
    "profound learning": "deep learning",
    "arbitrary woodland": "random forest",
}


def _mock_phrases():
    """Patch _load_phrases so tests don't depend on the data file."""
    return patch("snoopy.analysis.text_forensics._load_phrases", return_value=_TEST_PHRASES)


class TestTorturedPhrasesDetection:
    def test_multiple_tortured_phrases_flagged_suspicious(self):
        text = (
            "We applied a fake neural system to classify inputs, "
            "then used profound learning for feature extraction. "
            "Finally, an arbitrary woodland model was trained on the dataset."
        )
        with _mock_phrases():
            result = detect_tortured_phrases(text)
        assert isinstance(result, TorturedPhraseResult)
        assert result.suspicious is True
        assert result.match_count >= 3
        assert result.unique_phrases >= 3
        assert len(result.matches) >= 3


class TestTorturedPhrasesCleanText:
    def test_normal_academic_text_not_suspicious(self):
        text = (
            "We applied an artificial neural network to classify inputs, "
            "then used deep learning for feature extraction. "
            "Finally, a random forest model was trained on the dataset."
        )
        with _mock_phrases():
            result = detect_tortured_phrases(text)
        assert result.suspicious is False
        assert result.match_count == 0
        assert result.matches == []


class TestTorturedPhrasesSingleMatch:
    def test_single_match_below_default_threshold(self):
        text = (
            "The experiment used a fake neural system to process data. "
            "Results were analyzed using standard statistical methods."
        )
        with _mock_phrases():
            result = detect_tortured_phrases(text, min_matches=2)
        assert result.suspicious is False
        assert result.match_count == 1


class TestTorturedPhrasesMinMatchesConfig:
    def test_single_match_above_lowered_threshold(self):
        text = (
            "The experiment used a fake neural system to process data. "
            "Results were analyzed using standard statistical methods."
        )
        with _mock_phrases():
            result = detect_tortured_phrases(text, min_matches=1)
        assert result.suspicious is True
        assert result.match_count >= 1


class TestTorturedPhrasesEmptyText:
    def test_empty_string_not_suspicious(self):
        result = detect_tortured_phrases("")
        assert result.suspicious is False
        assert result.match_count == 0
        assert result.matches == []


class TestTorturedPhrasesCaseInsensitive:
    def test_capitalized_phrase_still_detected(self):
        text = (
            "We used a Fake Neural System and Profound Learning approach "
            "to solve the classification task."
        )
        with _mock_phrases():
            result = detect_tortured_phrases(text, min_matches=1)
        assert result.suspicious is True
        assert result.match_count >= 1
        found_phrases = [m.tortured_phrase.lower() for m in result.matches]
        assert "fake neural system" in found_phrases


class TestTorturedPhrasesMatchDetails:
    def test_match_fields_populated_correctly(self):
        text = (
            "We trained a fake neural system on the data. "
            "A profound learning framework was also evaluated. "
            "The arbitrary woodland classifier outperformed baselines."
        )
        with _mock_phrases():
            result = detect_tortured_phrases(text, min_matches=1)
        assert len(result.matches) >= 3
        for match in result.matches:
            assert isinstance(match, TorturedPhraseMatch)
            assert isinstance(match.tortured_phrase, str)
            assert len(match.tortured_phrase) > 0
            assert isinstance(match.correct_phrase, str)
            assert len(match.correct_phrase) > 0
            assert isinstance(match.position, int)
            assert match.position >= 0
            assert isinstance(match.context, str)
            assert len(match.context) > 0

        phrase_to_correct = {
            "fake neural system": "artificial neural network",
            "profound learning": "deep learning",
            "arbitrary woodland": "random forest",
        }
        for match in result.matches:
            key = match.tortured_phrase.lower()
            if key in phrase_to_correct:
                assert match.correct_phrase.lower() == phrase_to_correct[key]
