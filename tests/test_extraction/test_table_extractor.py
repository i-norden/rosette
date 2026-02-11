"""Tests for table extraction from PDFs."""

from __future__ import annotations

import pytest

from snoopy.extraction.table_extractor import (
    TableInfo,
    _clean_cell,
    _table_to_raw_text,
    extract_tables,
)


class TestCleanCell:
    def test_none_returns_empty_string(self) -> None:
        assert _clean_cell(None) == ""

    def test_strips_whitespace(self) -> None:
        assert _clean_cell("  hello  ") == "hello"

    def test_converts_int_to_string(self) -> None:
        assert _clean_cell(42) == "42"

    def test_converts_float_to_string(self) -> None:
        assert _clean_cell(3.14) == "3.14"

    def test_empty_string(self) -> None:
        assert _clean_cell("") == ""


class TestTableToRawText:
    def test_empty_table(self) -> None:
        result = _table_to_raw_text([], [])
        assert result == ""

    def test_headers_only(self) -> None:
        result = _table_to_raw_text(["Name", "Value"], [])
        assert "Name\tValue" in result
        assert "-" * 40 in result

    def test_full_table(self) -> None:
        headers = ["Col1", "Col2"]
        rows = [["a", "b"], ["c", "d"]]
        result = _table_to_raw_text(headers, rows)
        lines = result.split("\n")
        assert lines[0] == "Col1\tCol2"
        assert lines[1] == "-" * 40
        assert lines[2] == "a\tb"
        assert lines[3] == "c\td"

    def test_no_headers(self) -> None:
        result = _table_to_raw_text([], [["x", "y"]])
        assert result == "x\ty"


class TestExtractTables:
    def test_raises_file_not_found(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            extract_tables(str(tmp_path / "nonexistent.pdf"))

    def test_raises_runtime_error_on_bad_pdf(self, tmp_path, monkeypatch) -> None:
        import pdfplumber

        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"not a real PDF")

        def _raise(*args, **kwargs):
            raise Exception("parse error")

        monkeypatch.setattr(pdfplumber, "open", _raise)

        with pytest.raises(RuntimeError, match="Failed to open PDF"):
            extract_tables(str(bad_pdf))

    def test_returns_table_info_objects(self, tmp_path, monkeypatch) -> None:
        import pdfplumber

        class _FakePage:
            def extract_tables(self):
                return [
                    [["Header1", "Header2"], ["val1", "val2"], ["val3", "val4"]],
                ]

        class _FakePdf:
            pages = [_FakePage()]

            def close(self):
                pass

        pdf_path = tmp_path / "tables.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(pdfplumber, "open", lambda path: _FakePdf())

        tables = extract_tables(str(pdf_path))
        assert len(tables) == 1
        assert isinstance(tables[0], TableInfo)
        assert tables[0].page_number == 1
        assert tables[0].table_index == 0
        assert tables[0].headers == ["Header1", "Header2"]
        assert tables[0].rows == [["val1", "val2"], ["val3", "val4"]]
        assert "Header1" in tables[0].raw_text

    def test_skips_empty_tables(self, tmp_path, monkeypatch) -> None:
        import pdfplumber

        class _FakePage:
            def extract_tables(self):
                return [[], None, [["H1"]]]

        class _FakePdf:
            pages = [_FakePage()]

            def close(self):
                pass

        pdf_path = tmp_path / "sparse.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(pdfplumber, "open", lambda path: _FakePdf())

        tables = extract_tables(str(pdf_path))
        # Only the table with [["H1"]] should be parsed (headers only, no rows)
        assert len(tables) == 1
        assert tables[0].headers == ["H1"]
        assert tables[0].rows == []
