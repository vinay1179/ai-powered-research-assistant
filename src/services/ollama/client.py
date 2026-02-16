import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from src.config import Settings
from src.exceptions import OllamaConnectionError, OllamaException, OllamaTimeoutError
from src.schemas.ollama import RAGResponse
from src.services.ollama.prompts import RAGPromptBuilder, ResponseParser

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama local LLM service."""

    def __init__(self, settings: Settings):
        self.base_url = settings.ollama_host
        self.timeout = httpx.Timeout(float(settings.ollama_timeout))
        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()

    async def health_check(self) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/version")

                if response.status_code == 200:
                    version_data = response.json()
                    return {
                        "status": "healthy",
                        "message": "Ollama service is running",
                        "version": version_data.get("version", "unknown"),
                    }
                raise OllamaException(f"Ollama returned status {response.status_code}")

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Ollama health check failed: {str(e)}")

    async def list_models(self) -> List[Dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")

                if response.status_code == 200:
                    data = response.json()
                    return data.get("models", [])
                raise OllamaException(f"Failed to list models: {response.status_code}")

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error listing models: {e}")

    async def generate(self, model: str, prompt: str, stream: bool = False, **kwargs) -> Optional[Dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                data = {"model": model, "prompt": prompt, "stream": stream, **kwargs}

                response = await client.post(f"{self.base_url}/api/generate", json=data)

                if response.status_code == 200:
                    return response.json()
                raise OllamaException(f"Generation failed: {response.status_code}")

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error generating with Ollama: {e}")

    async def generate_stream(self, model: str, prompt: str, **kwargs):
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                data = {"model": model, "prompt": prompt, "stream": True, **kwargs}

                async with client.stream("POST", f"{self.base_url}/api/generate", json=data) as response:
                    if response.status_code != 200:
                        raise OllamaException(f"Streaming generation failed: {response.status_code}")

                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                chunk = json.loads(line)
                                yield chunk
                            except json.JSONDecodeError:
                                logger.warning("Failed to parse streaming chunk from Ollama")
                                continue

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error in streaming generation: {e}")

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str,
        use_structured_output: bool = True,
    ) -> Dict[str, Any]:
        try:
            if use_structured_output:
                prompt_data = self.prompt_builder.create_structured_prompt(query, chunks)
                response = await self.generate(
                    model=model,
                    prompt=prompt_data["prompt"],
                    temperature=0.7,
                    top_p=0.9,
                    format=prompt_data["format"],
                )
            else:
                prompt = self.prompt_builder.create_rag_prompt(query, chunks)
                response = await self.generate(
                    model=model,
                    prompt=prompt,
                    temperature=0.7,
                    top_p=0.9,
                    format="json",
                )

            if response and "response" in response:
                parsed_response = self.response_parser.parse_structured_response(response["response"])

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
            raise OllamaException("No response generated from Ollama")

        except Exception as e:
            logger.error(f"Error generating RAG answer: {e}")
            raise OllamaException(f"Failed to generate RAG answer: {e}")

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str,
    ):
        try:
            prompt = self.prompt_builder.create_rag_prompt(query, chunks)
            async for chunk in self.generate_stream(
                model=model,
                prompt=prompt,
                temperature=0.7,
                top_p=0.9,
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Error generating streaming RAG answer: {e}")
            raise OllamaException(f"Failed to generate streaming RAG answer: {e}")
