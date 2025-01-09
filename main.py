
####


from dataclasses import dataclass
import time
from typing import Callable

from sentence_transformers import SentenceTransformer

from cerebraai.algorithms.analysis import get_best_model
from cerebraai.models.llm import LLM, LLMConditions
from cerebraai.models.orchestrator import Orchestrator

from openai import OpenAI

def openai_executor(prompt: str) -> str:

    client = OpenAI(api_key="")

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