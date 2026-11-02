
import asyncio
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


class Embedding:

    def __init__(
        self,
        model_name: str = "gemini-embedding-001"
    ):
        self.model_name = model_name

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from environment")

        self.client = genai.Client(api_key=api_key)

    async def embed_chunks(
        self,
        chunks: list[str]
    ) -> list[list[float]]:

        if not chunks:
            return []

        result = await asyncio.to_thread(
            self.client.models.embed_content,
            model=self.model_name,
            contents=chunks,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768
            )
        )

        return [item.values for item in result.embeddings]

    async def embed_query(self, prompt: str) -> list[float]:

        result = await asyncio.to_thread(
            self.client.models.embed_content,
            model=self.model_name,
            contents=prompt,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768
            )
        )

        return result.embeddings[0].values


embedding = Embedding()
