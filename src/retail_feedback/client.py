"""Small adapter around the official OpenAI Python SDK."""

import os


class ModelAPIError(Exception):
    """An API failure that should stop an evaluation rather than count as a prediction."""


class OpenAIClient:
    def __init__(self, model: str):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Set OPENAI_API_KEY before running a live experiment")
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def complete(self, prompt: str) -> str:
        from openai import APIConnectionError, APIStatusError

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
        except APIStatusError as exc:
            code = getattr(exc, "code", None)
            if code is None and isinstance(getattr(exc, "body", None), dict):
                code = exc.body.get("code")
            if code == "credit_balance_exhausted":
                raise ModelAPIError(
                    "OpenAI API credits are exhausted for this key's organization. "
                    "Add credits at https://platform.openai.com/settings/organization/billing/ "
                    "and retry. No evaluation report was saved."
                ) from None
            if exc.status_code == 429:
                raise ModelAPIError("OpenAI API rate or usage limit reached. Check your API limits and retry later.") from None
            raise ModelAPIError(f"OpenAI API request failed (HTTP {exc.status_code}, code {code or 'unknown'}).") from None
        except APIConnectionError:
            raise ModelAPIError("Could not connect to the OpenAI API. Check your connection and retry.") from None
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Model returned an empty response")
        return content
