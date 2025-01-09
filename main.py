
####


from dataclasses import dataclass
import time
from typing import Callable

from sentence_transformers import SentenceTransformer

from src.algorithms.analysis import get_best_model
from src.models.llm import LLM, LLMConditions
from src.models.orchestrator import Orchestrator

from openai import OpenAI

def openai_executor(prompt: str) -> str:

    client = OpenAI(api_key="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")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


llms = [
    LLM("gpt-4", LLMConditions(domain="creativity", sentiment="positive", topic="arts", description="Optimized for reasoning, creativity, and complex tasks."), openai_executor),
    LLM("gpt-4o", LLMConditions(domain="general", sentiment="positive", topic="general", description="Moderately optimized for balanced tasks and cost-efficiency."), openai_executor),
    LLM("gpt-3.5", LLMConditions(domain="general", sentiment="positive", topic="general", description="Suitable for simple, straightforward tasks."), openai_executor),
]

orchestrator = Orchestrator(llms, SentenceTransformer("all-MiniLM-L12-v2"))

# Example usage
prompt = "Explain creativity"
result = orchestrator.execute(prompt)

print(result)