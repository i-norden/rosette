"""Anthropic Claude implementation of the LLMProvider interface."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path

import anthropic

from .base import LLMResponse

logger = logging.getLogger(__name__)

# Mapping of file extensions to MIME media types supported by the Claude
# vision API.
_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Default concurrency limit to avoid overwhelming the API.
_DEFAULT_MAX_CONCURRENT = 5

# Retry configuration.
_MAX_RETRIES = 3
_RETRY_BASE_DELAY_S = 1.0


def _detect_media_type(image_path: str) -> str:
    """Return the MIME type for *image_path* based on its file extension."""
    ext = Path(image_path).suffix.lower()
    media_type = _MEDIA_TYPES.get(ext)
    if media_type is None:
        raise ValueError(
            f"Unsupported image extension '{ext}'. "
            f"Supported: {', '.join(_MEDIA_TYPES)}"
        )
    return media_type


def _read_image_b64(image_path: str) -> str:
    """Read a file from disk and return its base-64 encoded content."""
    with open(image_path, "rb") as fh:
        return base64.standard_b64encode(fh.read()).decode("ascii")


def _try_parse_json(text: str) -> dict | None:
    """Attempt to extract and parse a JSON object from *text*.

    The model may wrap the JSON in a markdown code fence, so we try to
    strip that first before falling back to parsing the raw text.
    """
    cleaned = text.strip()

    # Strip optional markdown code fence.
    if cleaned.startswith("```"):
        # Remove opening fence (with optional language tag).
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3].strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None


class ClaudeProvider:
    """LLM provider backed by the Anthropic Claude API.

    Parameters:
        api_key: Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY``
            environment variable when *None*.
        default_model: Model identifier used when a request does not specify
            one explicitly.
        max_concurrent: Maximum number of concurrent API requests allowed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "claude-sonnet-4-5-20250929",
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "An Anthropic API key must be supplied either via the "
                "'api_key' parameter or the ANTHROPIC_API_KEY environment "
                "variable."
            )
        self.client = anthropic.AsyncAnthropic(api_key=resolved_key)
        self.default_model = default_model
        self._semaphore = asyncio.Semaphore(max_concurrent)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    async def _call_with_retry(
        self,
        messages: list[dict],
        model: str | None = None,
        system: str = "",
        max_tokens: int = 4096,
    ) -> anthropic.types.Message:
        """Send a request to the Claude API with exponential-backoff retry.

        Retries on rate-limit (429) and server errors (5xx) up to
        ``_MAX_RETRIES`` times.
        """
        model = model or self.default_model
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                async with self._semaphore:
                    return await self.client.messages.create(**kwargs)
            except anthropic.RateLimitError as exc:
                last_exc = exc
                delay = _RETRY_BASE_DELAY_S * (2 ** attempt)
                logger.warning(
                    "Rate-limited (attempt %d/%d). Retrying in %.1fs ...",
                    attempt + 1,
                    _MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
            except anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    last_exc = exc
                    delay = _RETRY_BASE_DELAY_S * (2 ** attempt)
                    logger.warning(
                        "Server error %d (attempt %d/%d). Retrying in %.1fs ...",
                        exc.status_code,
                        attempt + 1,
                        _MAX_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        # All retries exhausted -- re-raise the last exception.
        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _message_to_response(
        message: anthropic.types.Message,
        response_schema: dict | None = None,
    ) -> dict:
        """Convert an Anthropic ``Message`` to the standard response dict."""
        raw_content = message.content[0].text if message.content else ""
        parsed = _try_parse_json(raw_content) if response_schema else None
        return {
            "content": raw_content,
            "parsed": parsed,
            "model": message.model,
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        }

    # --------------------------------------------------------------------- #
    # Public API (satisfies LLMProvider protocol)
    # --------------------------------------------------------------------- #

    async def analyze_image(
        self,
        image_path: str,
        prompt: str,
        system: str = "",
        response_schema: dict | None = None,
    ) -> dict:
        """Analyze a single image with the given prompt.

        The image is base-64 encoded and sent inline as a vision content
        block alongside the user's text prompt.
        """
        media_type = _detect_media_type(image_path)
        image_data = _read_image_b64(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]

        response = await self._call_with_retry(messages, system=system)
        return self._message_to_response(response, response_schema)

    async def analyze_text(
        self,
        text: str,
        prompt: str,
        system: str = "",
        response_schema: dict | None = None,
    ) -> dict:
        """Analyze a text payload with the given prompt."""
        combined = f"{prompt}\n\n---\n\n{text}"
        messages = [
            {
                "role": "user",
                "content": combined,
            },
        ]

        response = await self._call_with_retry(messages, system=system)
        return self._message_to_response(response, response_schema)

    async def analyze_images_batch(
        self,
        requests: list[dict],
    ) -> list[dict]:
        """Process a batch of image-analysis requests.

        Currently processes requests sequentially, respecting the
        concurrency semaphore.  A future iteration may leverage the
        Anthropic batch API for additional cost savings.

        Each item in *requests* must contain at minimum ``image_path`` and
        ``prompt``.  Optional keys: ``system``, ``model``,
        ``response_schema``.
        """
        results: list[dict] = []
        for req in requests:
            image_path: str = req["image_path"]
            prompt: str = req["prompt"]
            system: str = req.get("system", "")
            response_schema: dict | None = req.get("response_schema")

            result = await self.analyze_image(
                image_path=image_path,
                prompt=prompt,
                system=system,
                response_schema=response_schema,
            )
            results.append(result)

        return results
