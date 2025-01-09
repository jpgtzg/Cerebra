# Written by Juan Pablo Gutiérrez
# 02/01/2025

from dataclasses import dataclass
import time
from typing import Callable

@dataclass
class LLMResponse:
    response: str
    llm_name: str
    status: dict
    execution_time: float

@dataclass
class LLMConditions:
    domain: str
    sentiment: str
    topic: str
    description: str

    def to_description(self):
        return f"Domain: {self.domain}, Sentiment: {self.sentiment}, Topic: {self.topic}, Description: {self.description}"

class LLMCallable(Callable):
    def __init__(self, model: str, api_key: str, conditions: LLMConditions, executor: Callable):
        self.model = model
        self.api_key = api_key
        self.conditions = conditions
        self.executor = executor

    def __call__(self, prompt: str) -> LLMResponse:
        return self.executor(prompt)

class LLM:

    model: str
    api_key: str
    conditions: LLMConditions
    executor: LLMCallable
    """
    A class to represent a Language Model (LLM).
    """

    def __init__(self, model: str, api_key: str, conditions: LLMConditions, executor: LLMCallable):
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
        start_time = time.time()
        response = self.executor(prompt)
        end_time = time.time()
        execution_time = end_time - start_time

        return LLMResponse(response=response, llm_name=self.model, status="success", execution_time=execution_time)

    def to_dict(self):
        return {
            "model": self.model,
            "conditions": self.conditions,
        }

