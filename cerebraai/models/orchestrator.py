# Written by Juan Pablo Gutiérrez
# 03/01/2025

from sentence_transformers import SentenceTransformer
from cerebraai.algorithms.analysis import get_best_model
from cerebraai.models.llm import LLM, LLMResponse

class Orchestrator:

    llms: list[LLM]
    analysis_weights: dict
    sentiment_weights: dict
    emotion_weights: dict

    """
    A class that represents an AI Orchestrator. It will balance prompting to different LLMs based on the user's prompt.
    """

    def __init__(self, llms: list[LLM], text_model: SentenceTransformer, analysis_weights: dict = {"semantic": 0.3, "topic": 0.5, "sentiment": 0.3}, sentiment_weights: dict = {"positive": 0.5, "negative": 0.5}, emotion_weights: dict = {"happy": 0.5, "sad": 0.5}):
        self.llms = llms
        self.text_model = text_model
        self.analysis_weights = analysis_weights
        self.sentiment_weights = sentiment_weights
        self.emotion_weights = emotion_weights

    def execute(self, prompt: str, analysis_weights: dict = None, sentiment_weights: dict = None, emotion_weights: dict = None) -> LLMResponse:
        analysis_weights = analysis_weights if analysis_weights is not None else self.analysis_weights
        sentiment_weights = sentiment_weights if sentiment_weights is not None else self.sentiment_weights
        emotion_weights = emotion_weights if emotion_weights is not None else self.emotion_weights
        
        llm = get_best_model(self.text_model, prompt, self.llms, analysis_weights, sentiment_weights, emotion_weights)["llm"]
        return llm.execute(prompt)

    def add_llm(self, llm: LLM):
        self.llms.append(llm)
