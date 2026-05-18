"""
Thin async client for the Mistral AI API.
Handles embeddings and chat completions with retry logic.
"""
from __future__ import annotations
 
import asyncio
import logging
from typing import Any
 
import httpx
 
from app.config import (
    MISTRAL_API_KEY,
    MISTRAL_BASE_URL,
    MISTRAL_CHAT_MODEL,
    MISTRAL_EMBED_MODEL,
    MAX_GENERATION_TOKENS,
    TEMPERATURE,
)
 
logger = logging.getLogger(__name__)
 
_client: httpx.AsyncClient | None = None
 
 
def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=MISTRAL_BASE_URL,
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
    return _client
 
 
async def close_client() -> None:
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None
 
 
async def _request_with_retry(
    method: str,
    url: str,
    json_body: dict[str, Any],
    max_retries: int = 3,
) -> dict[str, Any]:
    """Make an HTTP request with exponential backoff on rate-limit / server errors."""
    client = _get_client()
    for attempt in range(max_retries):
        try:
            resp = await client.request(method, url, json=json_body)
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = 2 ** attempt
                logger.warning(
                    "Mistral %s %s returned %d – retrying in %ds",
                    method, url, resp.status_code, wait,
                )
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError:
            raise
        except httpx.HTTPError as exc:
            if attempt == max_retries - 1:
                raise
            logger.warning("HTTP error %s – retry %d", exc, attempt + 1)
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError("Max retries exceeded")
 
 
# ──────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────
 
async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts via Mistral's embedding endpoint.
    The API accepts up to ~16 texts per call; we batch internally.
    """
    if not texts:
        return []
 
    all_embeddings: list[list[float]] = []
    batch_size = 16
 
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        data = await _request_with_retry(
            "POST",
            "/embeddings",
            {
                "model": MISTRAL_EMBED_MODEL,
                "input": batch,
            },
        )
        # Sort by index to preserve order
        items = sorted(data["data"], key=lambda x: x["index"])
        all_embeddings.extend([item["embedding"] for item in items])
 
    return all_embeddings
 
 
async def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    results = await embed_texts([text])
    return results[0]
 
 
# ──────────────────────────────────────────────
# Chat completion
# ──────────────────────────────────────────────
 
async def chat_completion(
    messages: list[dict[str, str]],
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_GENERATION_TOKENS,
    json_mode: bool = False,
) -> str:
    """Send a chat completion request and return the assistant's text."""
    payload: dict[str, Any] = {
        "model": MISTRAL_CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
 
    data = await _request_with_retry("POST", "/chat/completions", payload)
    return data["choices"][0]["message"]["content"]