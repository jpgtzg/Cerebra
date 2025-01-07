# Written by Juan Pablo Gutiérrez
# 05/01/2025
# This file handles analysis of a prompt.

from algorithms.semantic import get_semantic_based_model
from algorithms.sentiment import get_sentiment_based_model
from algorithms.topic import get_topic_based_model
from models.llm import LLM

def get_best_model(prompt: str, llms: LLM) -> str:
    """
    Gets the best model for the given prompt.

    Args:
        prompt (str): The prompt to analyze.
        llms (dict): The LLM models to decide on.

    Returns:
        str: The best model for the given prompt.
    """

    semantic_model = get_semantic_based_model(prompt, llms)
    topic_model = get_topic_based_model(prompt, llms)
    sentiment_model = get_sentiment_based_model(prompt, llms)

    return get_semantic_based_model(prompt, llms)