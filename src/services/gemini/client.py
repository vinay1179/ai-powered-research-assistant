import logging
from typing import Any, Dict, Optional

import httpx
from src.config import Settings
from src.services.ollama.prompts import RAGPromptBuilder, ResponseParser

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for interacting with Gemini API."""

    def __init__(self, settings: Settings):
        self.api_key = settings.gemini_api_key
        self.model = settings.gemini_model
        self.base_url = settings.gemini_base_url.rstrip("/")
        self.timeout = httpx.Timeout(settings.gemini_timeout)
        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()

    async def generate(self, prompt: str, response_mime_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            raise ValueError("Gemini API key is not configured")

        url = f"{self.base_url}/models/{self.model}:generateContent"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
        }
        if response_mime_type:
            payload["generationConfig"] = {"responseMimeType": response_mime_type}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, params={"key": self.api_key}, json=payload)
            if response.status_code != 200:
                raise RuntimeError(f"Gemini generation failed: {response.status_code}")
            return response.json()

    @staticmethod
    def _extract_text(payload: Dict[str, Any]) -> Optional[str]:
        try:
            candidates = payload.get("candidates") or []
            if not candidates:
                return None
            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            if not parts:
                return None
            return parts[0].get("text")
        except Exception:
            return None

    async def generate_rag_answer(self, query: str, chunks: list[dict[str, Any]]) -> Dict[str, Any]:
        prompt = self.prompt_builder.create_rag_prompt(query, chunks)
        response = await self.generate(prompt, response_mime_type="application/json")
        if not response:
            raise RuntimeError("Gemini returned empty response")

        text = self._extract_text(response)
        if not text:
            raise RuntimeError("Gemini response did not include text content")

        parsed_response = self.response_parser.parse_structured_response(text)

        if not parsed_response.get("sources"):
            sources = []
            seen_urls = set()
            for chunk in chunks:
                arxiv_id = chunk.get("arxiv_id")
                if arxiv_id:
                    arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
                    if pdf_url not in seen_urls:
                        sources.append(pdf_url)
                        seen_urls.add(pdf_url)
            parsed_response["sources"] = sources

        if not parsed_response.get("citations"):
            citations = list({chunk.get("arxiv_id") for chunk in chunks if chunk.get("arxiv_id")})
            parsed_response["citations"] = citations[:5]

        return parsed_response
