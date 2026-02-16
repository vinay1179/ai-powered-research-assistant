import json
import logging
from typing import Iterator

import gradio as gr
import httpx

logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000/api/v1"
DEFAULT_MODEL = "llama3.2:1b"
AVAILABLE_CATEGORIES = ["cs.AI", "cs.LG"]


async def stream_response(
    query: str,
    top_k: int = 3,
    use_hybrid: bool = True,
    provider: str = "ollama",
    model: str = DEFAULT_MODEL,
    categories: str = "",
) -> Iterator[str]:
    if not query.strip():
        yield "Please enter a question."
        return

    category_list = [cat.strip() for cat in categories.split(",") if cat.strip()] if categories else None
    payload = {
        "query": query,
        "top_k": top_k,
        "use_hybrid": use_hybrid,
        "model": model,
        "provider": provider,
        "categories": category_list,
    }

    try:
        if provider == "gemini":
            url = f"{API_BASE_URL}/ask"
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    yield f"Error: API returned status {response.status_code}"
                    return
                data = response.json()
                answer = data.get("answer", "No answer returned.")
                sources = data.get("sources", [])
                chunks_used = data.get("chunks_used", 0)
                search_mode = data.get("search_mode", "unknown")
                formatted = _format_answer(answer, sources, chunks_used, search_mode)
                yield formatted
                return

        url = f"{API_BASE_URL}/stream"
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload, headers={"Accept": "text/plain"}) as response:
                if response.status_code != 200:
                    yield f"Error: API returned status {response.status_code}"
                    return

                current_answer = ""
                sources = []
                chunks_used = 0
                search_mode = ""

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                            if "error" in data:
                                yield f"Error: {data['error']}"
                                return

                            if "sources" in data:
                                sources = data["sources"]
                                chunks_used = data.get("chunks_used", 0)
                                search_mode = data.get("search_mode", "unknown")
                                continue

                            if "chunk" in data:
                                current_answer += data["chunk"]
                                yield _format_answer(current_answer, sources, chunks_used, search_mode)

                            if data.get("done", False):
                                final_answer = data.get("answer", current_answer)
                                yield _format_answer(final_answer, sources, chunks_used, search_mode)
                                break

                        except json.JSONDecodeError:
                            continue

    except httpx.RequestError as e:
        yield f"Connection error: {str(e)}\nMake sure the API server is running at {API_BASE_URL}"
    except Exception as e:
        yield f"Unexpected error: {str(e)}"


def _format_answer(answer: str, sources: list[str], chunks_used: int, search_mode: str) -> str:
    formatted = answer
    if sources or chunks_used:
        formatted += "\n\nSearch Info:\n"
        formatted += f"- Mode: {search_mode}\n"
        formatted += f"- Chunks used: {chunks_used}\n"
        if sources:
            formatted += f"- Sources: {len(sources)} papers\n"
            for i, source in enumerate(sources[:3], 1):
                formatted += f"  {i}. [{source.split('/')[-1]}]({source})\n"
            if len(sources) > 3:
                formatted += f"  ... and {len(sources) - 3} more\n"
    return formatted


def create_gradio_interface():
    with gr.Blocks(
        title="arXiv Paper Curator - RAG Chat",
        theme=gr.themes.Soft(),
    ) as interface:
        gr.Markdown(
            """
            # arXiv Paper Curator - RAG Chat

            Ask questions about machine learning and AI research papers from arXiv.
            The system will search through indexed papers and provide answers with sources.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What are transformers in machine learning?",
                    lines=2,
                    max_lines=5,
                )

            with gr.Column(scale=1):
                submit_btn = gr.Button("Ask Question", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                with gr.Accordion("Advanced Options", open=False):
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Number of chunks to retrieve",
                        info="More chunks = more context but slower generation",
                    )

                    use_hybrid = gr.Checkbox(
                        value=True,
                        label="Use hybrid search (BM25 + vector embeddings)",
                        info="Usually better results than keyword-only search",
                    )

                    provider_choice = gr.Dropdown(
                        choices=["ollama", "gemini"],
                        value="ollama",
                        label="LLM Provider",
                        info="Ollama streams tokens; Gemini returns once",
                    )

                    model_choice = gr.Dropdown(
                        choices=["llama3.2:1b", "llama3.2:3b", "llama3.1:8b", "qwen2.5:7b"],
                        value=DEFAULT_MODEL,
                        label="Ollama Model",
                        info="Used only when provider is Ollama",
                    )

                    categories = gr.Textbox(
                        label="arXiv Categories (optional)",
                        placeholder="cs.AI, cs.LG, cs.CL",
                        info="Comma-separated. Leave empty for all categories",
                    )

        response_output = gr.Markdown(
            label="Answer",
            value="Ask a question to get started!",
            height=400,
            elem_classes=["response-markdown"],
        )

        gr.Examples(
            examples=[
                ["What are transformers in machine learning?", 3, True, "ollama", "llama3.2:1b", "cs.AI, cs.LG"],
                ["How do convolutional neural networks work?", 5, True, "ollama", "llama3.2:1b", "cs.CV, cs.LG"],
                ["What is attention mechanism in deep learning?", 4, False, "gemini", "llama3.2:1b", "cs.AI"],
                ["Explain reinforcement learning algorithms", 3, True, "ollama", "llama3.2:1b", "cs.LG, cs.AI"],
                ["What are the latest developments in NLP?", 5, True, "gemini", "llama3.2:1b", "cs.CL"],
            ],
            inputs=[query_input, top_k, use_hybrid, provider_choice, model_choice, categories],
        )

        submit_btn.click(
            fn=stream_response,
            inputs=[query_input, top_k, use_hybrid, provider_choice, model_choice, categories],
            outputs=[response_output],
            show_progress=True,
        )

        query_input.submit(
            fn=stream_response,
            inputs=[query_input, top_k, use_hybrid, provider_choice, model_choice, categories],
            outputs=[response_output],
            show_progress=True,
        )

        gr.Markdown(
            """
            ---

            Note: Make sure the API server is running at `http://localhost:8000` before using this interface.
            """
        )

    return interface


def main():
    print("Starting arXiv Paper Curator Gradio Interface...")
    print(f"API Base URL: {API_BASE_URL}")

    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()
