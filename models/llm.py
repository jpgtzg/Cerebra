# Written by Juan Pablo Gutiérrez
# 02/01/2025

from dataclasses import dataclass
from typing import Callable

@dataclass
class LLMResponse:
    response: str
    llm_name: str
    status: dict
    execution_time: float

class LLM:

    model: str
    api_key: str
    conditions: dict
    executor: Callable
    """
    A class to represent a Language Model (LLM).
    """

    def __init__(self, model: str, api_key: str, conditions: dict, executor: Callable):
        """
        Initializes the LLM with the given parameters.

        :param name: The name of the LLM.
        :param model: The model identifier.
        :param api_key: The API key for accessing the model.
        :param conditions: A dictionary of conditions for the model to be ran.
        :param executor: A callable to execute the model with a prompt.
        """
        self.model = model
        self.api_key = api_key
        self.conditions = conditions
        self.executor = executor

    def execute(self, prompt: str) -> LLMResponse:
        """
        Executes the model with the given prompt.

        :param prompt: The prompt to send to the model.
        :return: An LLMResponse containing the model's response.
        """
        return self.executor(prompt) if self.executor else None


