# Written by Juan Pablo Gutiérrez
# 06/01/2025
# This file handles semantic similarity between a prompt and a LLM model.

from sentence_transformers import SentenceTransformer
from algorithms.utils import similarity_score

def get_semantic_based_model(model: SentenceTransformer, prompt: str, llms: dict) -> str:
    """
    Gets the best semantic based model for the given prompt.

    Args:
        model (SentenceTransformer): The semantic model to use.
        prompt (str): The prompt to analyze.
        llms (dict): The LLM models to decide on.

    Returns:
        str: The best semantic based model.
    """
    similarities = {model_name: similarity_score(model, prompt, llms[model_name]) for model_name in llms.keys()}
    best_match = max(similarities, key=similarities.get)

    return {
        "best_match": best_match,
        "similarity": similarities[best_match]
    }
