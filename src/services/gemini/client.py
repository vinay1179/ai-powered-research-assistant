import logging
from typing import Any, Dict, Optional

import httpx
from src.config import Settings

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for interacting with Gemini API."""

    def __init__(self, settings: Settings):
        self.api_key = settings.gemini_api_key
        self.model = settings.gemini_model
        self.base_url = settings.gemini_base_url.rstrip("/")
        self.timeout = httpx.Timeout(settings.gemini_timeout)

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
