# Written by Juan Pablo Gutiérrez
# 05/01/2025
# This file handles analysis of a prompt.

from algorithms.semantic import get_semantic_based_model
from algorithms.topic import get_topic_based_model

def get_best_model(prompt: str, llms: dict) -> str:
    """
    Gets the best model for the given prompt.

    Args:
        prompt (str): The prompt to analyze.
        llms (dict): The LLM models to decide on.

    Returns:
        str: The best model for the given prompt.
    """
    return get_semantic_based_model(prompt, llms)