# Written by Juan Pablo Gutiérrez
# 03/01/2025

from sentence_transformers import SentenceTransformer
from cerebraai.algorithms.analysis import get_best_model
from cerebraai.models.llm import LLM, LLMResponse

class Orchestrator:

    llms: list[LLM]
    """
    A class that represents an AI Orchestrator. It will balance prompting to different LLMs based on the user's prompt.
    """

    def __init__(self, llms: list[LLM], text_model: SentenceTransformer):
        self.llms = llms
        self.text_model = text_model

    def execute(self, prompt: str, weights: dict = {"semantic": 0.3, "topic": 0.5, "sentiment": 0.3}, sentiment_weights: dict = {"positive": 0.5, "negative": 0.5}, emotion_weights: dict = {"happy": 0.5, "sad": 0.5}) -> LLMResponse:
        llm = get_best_model(self.text_model, prompt, self.llms, weights, sentiment_weights, emotion_weights)["llm"]
        return llm.execute(prompt)

    def add_llm(self, llm: LLM):
        self.llms.append(llm)
