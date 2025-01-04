# Written by Juan Pablo Gutiérrez
# 03/01/2025

from models.llm import LLM, LLMResponse

class Orchestrator:

    llms: list[LLM]
    """
    A class that represents an AI Orchestrator. It will balance prompting to different LLMs based on the user's prompt.
    """

    def __init__(self, llms: list[LLM]):
        self.llms = llms

    def execute(self, prompt: str) -> LLMResponse:
        llm = algorithms.balance_prompting(prompt, self.llms)
        return llm.execute(prompt)