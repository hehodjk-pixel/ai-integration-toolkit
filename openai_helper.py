import openai
import os
import json
import time
from typing import Optional, Generator
from dataclasses import dataclass

@dataclass
class AIResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float

# Approximate cost per 1K tokens (update as pricing changes)
COST_PER_1K = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}

class OpenAIHelper:
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def _calc_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = COST_PER_1K.get(model, {"input": 0.005, "output": 0.015})
        return (prompt_tokens / 1000 * pricing["input"]) + (completion_tokens / 1000 * pricing["output"])

    def chat(self, prompt: str, system: str = "You are a helpful assistant.",
             temperature: float = 0.7, max_tokens: int = 1000) -> AIResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        usage = response.usage
        cost = self._calc_cost(self.model, usage.prompt_tokens, usage.completion_tokens)
        return AIResponse(
            content=response.choices[0].message.content,
            model=self.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=round(cost, 6)
        )

    def stream(self, prompt: str, system: str = "You are a helpful assistant.") -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def embed(self, text: str, model: str = "text-embedding-3-small") -> list[float]:
        response = self.client.embeddings.create(input=text, model=model)
        return response.data[0].embedding

    def embed_many(self, texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
        response = self.client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]

    def classify(self, text: str, categories: list[str]) -> str:
        prompt = f"""Classify the following text into exactly one of these categories: {", ".join(categories)}

Text: {text}

Respond with only the category name, nothing else."""
        result = self.chat(prompt, temperature=0)
        return result.content.strip()

    def extract_json(self, text: str, schema: dict) -> dict:
        prompt = f"""Extract structured data from the text below and return valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Text: {text}

Return only the JSON object, no explanation."""
        result = self.chat(prompt, temperature=0)
        return json.loads(result.content.strip())

    def summarize(self, text: str, max_words: int = 100) -> str:
        prompt = f"Summarize the following text in under {max_words} words:\n\n{text}"
        return self.chat(prompt).content

    def translate(self, text: str, target_language: str) -> str:
        prompt = f"Translate the following text to {target_language}. Return only the translation:\n\n{text}"
        return self.chat(prompt, temperature=0).content


if __name__ == "__main__":
    ai = OpenAIHelper()

    # Basic chat
    response = ai.chat("What is machine learning in one sentence?")
    print(f"Response: {response.content}")
    print(f"Tokens used: {response.total_tokens} | Cost: ${response.cost_usd}")

    # Streaming
    print("\nStreaming response: ", end="")
    for chunk in ai.stream("List 3 Python libraries for data science."):
        print(chunk, end="", flush=True)
    print()
