import json
import logging
from typing import Any, Dict, List, Optional

from src.services.gemini.client import GeminiClient

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:
                return None
    return None


def _extract_gemini_text(response: Dict[str, Any]) -> Optional[str]:
    try:
        return GeminiClient._extract_text(response)
    except Exception:
        return None


async def call_llm_json(runtime, prompt: str) -> Optional[Dict[str, Any]]:
    settings = runtime.context.settings
    if settings.llm_provider == "gemini":
        response = await runtime.context.gemini_client.generate(
            prompt, response_mime_type="application/json"
        )
        return _extract_json(_extract_gemini_text(response) or "")

    response = await runtime.context.ollama_client.generate(
        model=settings.ollama_default_model, prompt=prompt, format="json"
    )
    raw = response.get("response") if isinstance(response, dict) else None
    return _extract_json(raw or "")


async def call_llm_text(runtime, prompt: str) -> str:
    settings = runtime.context.settings
    if settings.llm_provider == "gemini":
        response = await runtime.context.gemini_client.generate(prompt)
        return _extract_gemini_text(response) or ""

    response = await runtime.context.ollama_client.generate(
        model=settings.ollama_default_model, prompt=prompt
    )
    if isinstance(response, dict):
        return response.get("response") or ""
    return ""


def build_snippets(documents: List[dict], limit: int = 3) -> str:
    snippets = []
    for doc in documents[:limit]:
        chunk_text = doc.get("chunk_text") or doc.get("abstract") or ""
        preview = chunk_text[:400] + ("..." if len(chunk_text) > 400 else "")
        snippets.append(f"- {preview}")
    return "\n".join(snippets) if snippets else "No snippets available."


def build_context(documents: List[dict], max_chars: int = 4000) -> str:
    parts = []
    total = 0
    for doc in documents:
        chunk_text = doc.get("chunk_text") or doc.get("abstract") or ""
        if not chunk_text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        segment = chunk_text[:remaining]
        parts.append(segment)
        total += len(segment)
    return "\n\n".join(parts)
