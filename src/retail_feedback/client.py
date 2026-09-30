"""Small adapter around the official OpenAI Python SDK."""

import os


class OpenAIClient:
    def __init__(self, model: str):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Set OPENAI_API_KEY before running a live experiment")
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Model returned an empty response")
        return content
