
####


from dataclasses import dataclass
import time
from typing import Callable

from sentence_transformers import SentenceTransformer

from src.algorithms.analysis import get_best_model
from src.models.llm import LLM, LLMConditions
from src.models.orchestrator import Orchestrator

callable = lambda prompt: "Hello"

llms = [
    LLM("gpt-4", "sk-proj-123", LLMConditions(domain="creativity", sentiment="positive", topic="arts", description="Optimized for reasoning, creativity, and complex tasks."), callable),
    LLM("gpt-4o", "sk-proj-123", LLMConditions(domain="general", sentiment="positive", topic="general", description="Moderately optimized for balanced tasks and cost-efficiency."), callable),
    LLM("gpt-3.5", "sk-proj-123", LLMConditions(domain="general", sentiment="positive", topic="general", description="Suitable for simple, straightforward tasks."), callable),
]

orchestrator = Orchestrator(llms, SentenceTransformer("all-MiniLM-L12-v2"))

# Example usage
prompt = "What's a book"
result = orchestrator.execute(prompt)

print(result)