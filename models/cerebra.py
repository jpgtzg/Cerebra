"""
Written by Juan Pablo Gutiérrez
02/01/2025

Cerebra is a framework for creating and executing LLM agents based on the Orchestrator pattern and the user's prompt.

"""

from typing import Callable
from models.llm import LLM, LLMResponse
from models.orchestrator import Orchestrator

class Cerebra:

    llms: list[LLM]

    def __init__(self):
        self.llms = []
        self.agents = []

    def define_llm(self, model: str, api_key: str, conditions: dict, executor: Callable) -> LLM:
        llm = LLM(model, api_key, conditions, executor)
        self.llms.append(llm)
        return llm

    def define_orchestrator(self, llms: list[LLM]) -> Orchestrator:
        orchestrator = Orchestrator(llms)
        return orchestrator
