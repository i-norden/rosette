"""Tests for snoopy.validation module."""

from __future__ import annotations

import pytest

from snoopy.validation import validate_doi


class TestValidateDoi:
    """Tests for the validate_doi function."""

    def test_valid_doi(self):
        assert validate_doi("10.1234/abc.def") == "10.1234/abc.def"

    def test_valid_doi_complex(self):
        assert validate_doi("10.1000/journal.pone.0000001") == "10.1000/journal.pone.0000001"

    def test_strips_https_prefix(self):
        assert validate_doi("https://doi.org/10.1234/abc") == "10.1234/abc"

    def test_strips_http_prefix(self):
        assert validate_doi("http://doi.org/10.1234/abc") == "10.1234/abc"

    def test_strips_dx_prefix(self):
        assert validate_doi("http://dx.doi.org/10.1234/abc") == "10.1234/abc"

    def test_strips_doi_prefix_lowercase(self):
        assert validate_doi("doi:10.1234/abc") == "10.1234/abc"

    def test_strips_doi_prefix_uppercase(self):
        assert validate_doi("DOI:10.1234/abc") == "10.1234/abc"

    def test_strips_whitespace(self):
        assert validate_doi("  10.1234/abc  ") == "10.1234/abc"

    def test_strips_prefix_and_trailing_whitespace(self):
        assert validate_doi("https://doi.org/10.1234/abc  ") == "10.1234/abc"

    def test_invalid_no_slash(self):
        with pytest.raises(ValueError, match="Invalid DOI format"):
            validate_doi("10.1234")

    def test_invalid_wrong_prefix(self):
        with pytest.raises(ValueError, match="Invalid DOI format"):
            validate_doi("11.1234/abc")

    def test_invalid_empty(self):
        with pytest.raises(ValueError, match="Invalid DOI format"):
            validate_doi("")

    def test_invalid_random_string(self):
        with pytest.raises(ValueError, match="Invalid DOI format"):
            validate_doi("not-a-doi")

    def test_valid_long_registrant(self):
        assert validate_doi("10.12345/abc") == "10.12345/abc"

    def test_invalid_short_registrant(self):
        with pytest.raises(ValueError, match="Invalid DOI format"):
            validate_doi("10.123/abc")

    def test_rejects_overly_long_doi(self):
        """DOI exceeding 256 chars should raise ValueError."""
        long_doi = "10.1234/" + "a" * 300
        with pytest.raises(ValueError, match="too long"):
            validate_doi(long_doi)

    def test_rejects_shell_metacharacters(self):
        """DOI with shell metacharacters should be rejected."""
        with pytest.raises(ValueError, match="Invalid DOI format"):
            validate_doi("10.1234/abc;rm -rf /")

    def test_rejects_backtick(self):
        """DOI with backticks should be rejected."""
        with pytest.raises(ValueError, match="Invalid DOI format"):
            validate_doi("10.1234/abc`whoami`")

    def test_allows_parens_and_dashes(self):
        """DOI with parentheses and dashes should be valid."""
        assert validate_doi("10.1234/abc-def(1)") == "10.1234/abc-def(1)"

    def test_allows_underscores(self):
        """DOI with underscores should be valid."""
        assert validate_doi("10.1234/abc_def") == "10.1234/abc_def"

    def test_rejects_registrant_too_many_digits(self):
        """DOI registrant code with >9 digits should be rejected."""
        with pytest.raises(ValueError, match="Invalid DOI format"):
            validate_doi("10.1234567890/abc")

    def test_allows_registrant_9_digits(self):
        """DOI registrant code with exactly 9 digits should be valid."""
        assert validate_doi("10.123456789/abc") == "10.123456789/abc"
