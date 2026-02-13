"""Tests for the Claude LLM provider."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from snoopy.llm.claude import (
    ClaudeProvider,
    _detect_media_type,
    _read_image_b64,
    _try_parse_json,
)


class TestDetectMediaType:
    def test_png(self):
        assert _detect_media_type("image.png") == "image/png"

    def test_jpg(self):
        assert _detect_media_type("image.jpg") == "image/jpeg"

    def test_jpeg(self):
        assert _detect_media_type("image.jpeg") == "image/jpeg"

    def test_gif(self):
        assert _detect_media_type("image.gif") == "image/gif"

    def test_webp(self):
        assert _detect_media_type("image.webp") == "image/webp"

    def test_case_insensitive(self):
        assert _detect_media_type("image.PNG") == "image/png"
        assert _detect_media_type("image.JPG") == "image/jpeg"

    def test_unsupported_extension(self):
        with pytest.raises(ValueError, match="Unsupported image extension"):
            _detect_media_type("image.bmp")

    def test_path_with_directories(self):
        assert _detect_media_type("/data/figures/fig1.png") == "image/png"


class TestReadImageB64:
    def test_reads_and_encodes(self, tmp_path):
        img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        path = str(tmp_path / "test.png")
        img.save(path)
        b64 = _read_image_b64(path)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            _read_image_b64("/nonexistent/path.png")


class TestTryParseJson:
    def test_parses_valid_json(self):
        result = _try_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parses_json_with_code_fence(self):
        result = _try_parse_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_parses_json_with_generic_fence(self):
        result = _try_parse_json('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_returns_none_for_invalid_json(self):
        result = _try_parse_json("not json at all")
        assert result is None

    def test_returns_none_for_empty(self):
        result = _try_parse_json("")
        assert result is None

    def test_parses_with_whitespace(self):
        result = _try_parse_json('  \n  {"key": "value"}  \n  ')
        assert result == {"key": "value"}

    def test_nested_json(self):
        result = _try_parse_json('{"a": {"b": [1, 2, 3]}}')
        assert result == {"a": {"b": [1, 2, 3]}}


class TestClaudeProviderInit:
    def test_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove ANTHROPIC_API_KEY if present
            env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="API key must be supplied"):
                    ClaudeProvider(api_key=None)

    def test_accepts_explicit_key(self):
        with patch("anthropic.AsyncAnthropic"):
            provider = ClaudeProvider(api_key="sk-ant-test-key")
        assert provider.default_model == "claude-sonnet-4-5-20250929"

    def test_reads_key_from_env(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-env-key"}):
            with patch("anthropic.AsyncAnthropic"):
                provider = ClaudeProvider()
            assert provider.default_model == "claude-sonnet-4-5-20250929"

    def test_custom_model(self):
        with patch("anthropic.AsyncAnthropic"):
            provider = ClaudeProvider(
                api_key="sk-ant-test", default_model="claude-haiku-4-5-20251001"
            )
        assert provider.default_model == "claude-haiku-4-5-20251001"


class TestClaudeProviderRetry:
    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self):
        import anthropic as anthropic_mod

        with patch("anthropic.AsyncAnthropic"):
            provider = ClaudeProvider(api_key="sk-ant-test")

            # First call raises rate limit, second succeeds
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text='{"result": "ok"}')]
            mock_message.model = "claude-sonnet-4-5-20250929"
            mock_message.usage = MagicMock(input_tokens=100, output_tokens=50)

            mock_create = AsyncMock(
                side_effect=[
                    anthropic_mod.RateLimitError(
                        message="rate limited",
                        response=MagicMock(status_code=429),
                        body=None,
                    ),
                    mock_message,
                ]
            )
            provider.client.messages.create = mock_create

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await provider.analyze_text(text="test", prompt="analyze this")

            assert result["content"] == '{"result": "ok"}'
            assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_server_error(self):
        import anthropic as anthropic_mod

        with patch("anthropic.AsyncAnthropic"):
            provider = ClaudeProvider(api_key="sk-ant-test")

            mock_message = MagicMock()
            mock_message.content = [MagicMock(text="ok")]
            mock_message.model = "claude-sonnet-4-5-20250929"
            mock_message.usage = MagicMock(input_tokens=10, output_tokens=5)

            mock_create = AsyncMock(
                side_effect=[
                    anthropic_mod.APIStatusError(
                        message="bad gateway",
                        response=MagicMock(status_code=502),
                        body=None,
                    ),
                    mock_message,
                ]
            )
            provider.client.messages.create = mock_create

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await provider.analyze_text(text="test", prompt="analyze this")

            assert result["content"] == "ok"

    @pytest.mark.asyncio
    async def test_does_not_retry_client_error(self):
        import anthropic as anthropic_mod

        with patch("anthropic.AsyncAnthropic"):
            provider = ClaudeProvider(api_key="sk-ant-test")

            mock_create = AsyncMock(
                side_effect=anthropic_mod.APIStatusError(
                    message="bad request",
                    response=MagicMock(status_code=400),
                    body=None,
                )
            )
            provider.client.messages.create = mock_create

            with pytest.raises(anthropic_mod.APIStatusError):
                await provider.analyze_text(text="test", prompt="analyze")

            assert mock_create.call_count == 1


class TestClaudeProviderAnalyze:
    @pytest.mark.asyncio
    async def test_analyze_text_response(self):
        with patch("anthropic.AsyncAnthropic"):
            provider = ClaudeProvider(api_key="sk-ant-test")

            mock_message = MagicMock()
            mock_message.content = [MagicMock(text='{"finding": "suspicious"}')]
            mock_message.model = "claude-sonnet-4-5-20250929"
            mock_message.usage = MagicMock(input_tokens=100, output_tokens=50)

            provider.client.messages.create = AsyncMock(return_value=mock_message)

            result = await provider.analyze_text(
                text="some paper text",
                prompt="analyze this text",
                response_schema={"type": "object"},
            )

        assert result["content"] == '{"finding": "suspicious"}'
        assert result["parsed"] == {"finding": "suspicious"}
        assert result["model"] == "claude-sonnet-4-5-20250929"
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50

    @pytest.mark.asyncio
    async def test_analyze_image_sends_correct_format(self, tmp_path):
        img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        path = str(tmp_path / "test.png")
        img.save(path)

        with patch("anthropic.AsyncAnthropic"):
            provider = ClaudeProvider(api_key="sk-ant-test")

            mock_message = MagicMock()
            mock_message.content = [MagicMock(text="analysis result")]
            mock_message.model = "claude-sonnet-4-5-20250929"
            mock_message.usage = MagicMock(input_tokens=500, output_tokens=100)

            provider.client.messages.create = AsyncMock(return_value=mock_message)

            result = await provider.analyze_image(image_path=path, prompt="describe this image")

        assert result["content"] == "analysis result"
        # Verify the messages were constructed with image content
        call_args = provider.client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["type"] == "image"
        assert messages[0]["content"][0]["source"]["media_type"] == "image/png"
        assert messages[0]["content"][1]["type"] == "text"

    @pytest.mark.asyncio
    async def test_analyze_images_batch(self, tmp_path):
        img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        path = str(tmp_path / "test.png")
        img.save(path)

        with patch("anthropic.AsyncAnthropic"):
            provider = ClaudeProvider(api_key="sk-ant-test")

            mock_message = MagicMock()
            mock_message.content = [MagicMock(text="batch result")]
            mock_message.model = "claude-sonnet-4-5-20250929"
            mock_message.usage = MagicMock(input_tokens=100, output_tokens=50)

            provider.client.messages.create = AsyncMock(return_value=mock_message)

            results = await provider.analyze_images_batch(
                [
                    {"image_path": path, "prompt": "analyze 1"},
                    {"image_path": path, "prompt": "analyze 2"},
                ]
            )

        assert len(results) == 2
        assert all(r["content"] == "batch result" for r in results)
