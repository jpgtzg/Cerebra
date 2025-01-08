# Written by Juan Pablo Gutiérrez
# 06/01/2025
# This file handles sentiment analysis of a prompt.

from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Literal, Dict, Optional

def get_sentiment_analysis(prompt: str) -> dict:
    """
    Gets the sentiment analysis of the given prompt.

    Args:
        prompt (str): The prompt to analyze.

    Returns:
        dict: The sentiment scores of the prompt.
    """
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(prompt)

def get_sentiment_based_model(
    prompt: str,
    llms: Dict,
    method: Literal["dominant", "weighted"] = "dominant",
    weights: Optional[Dict] = None
) -> dict:
    """
    Gets the best sentiment-based model for the given prompt.

    Args:
        prompt (str): The prompt to analyze.
        llms (dict): The LLM models to decide on.
        method (Literal["dominant", "weighted"]): The method to use to decide on the model.
        weights (dict): The weights to use for the weighted method.
    Returns:
        dict: The best sentiment-based model and its score for the given prompt.
    """
    sentiment = get_sentiment_analysis(prompt)

    if method == "dominant":
        return get_dominant_sentiment_model(sentiment, llms)
    elif method == "weighted":
        if weights is None:
            raise ValueError("Weights parameter is required for the weighted method.")
        return get_weighted_sentiment_model(prompt, llms, weights)
    else:
        raise ValueError(f"Invalid method: {method}")
    
def get_dominant_sentiment_model(sentiment: dict, llms: dict) -> str:
    """
    Gets the dominant sentiment model from the sentiment analysis.
    """

    for llm in llms:
        llm_sentiment = get_sentiment_analysis(llm)
        if llm_sentiment["compound"] > sentiment["compound"]:
            sentiment = llm_sentiment

    dominant_sentiment = max(sentiment, key=lambda x: sentiment[x] if x != "compound" else -1)

    selected_model = llms.get(dominant_sentiment, "gpt-3.5")  # Default to gpt-3.5 if no match

    return {
        "llm": selected_model,
        "score": sentiment[dominant_sentiment]
    }

def get_weighted_sentiment_model(prompt: str, llms: dict, weights: dict) -> dict:
    """
    Selects the best model based on weighted sentiment scores.

    Args:
        prompt (str): The prompt to analyze.
        llms (dict): The LLM models to decide on.
        weights (dict): Weights for positive, negative, and neutral sentiments.

    Returns:
        dict: The best model and its weighted score.
    """
    sentiment = get_sentiment_analysis(prompt)

    # Calculate weighted score for each model
    weighted_scores = {}
    for model, description in llms.items():
        weighted_score = sum(sentiment[sentiment_key] * weights[sentiment_key] for sentiment_key in weights.keys())
        weighted_scores[model] = weighted_score

    # Select the model with the highest weighted score
    best_model = max(weighted_scores, key=weighted_scores.get)

    return {
        "llm": best_model,
        "weighted_score": weighted_scores[best_model]
    }

# Define LLMs
llms = {
    "gpt-4": "Optimized for reasoning, creativity, and complex tasks.",
    "gpt-4o": "Moderately optimized for balanced tasks and cost-efficiency.",
    "gpt-3.5": "Suitable for simple, straightforward tasks."
}

# Define weights for sentiment categories
weights = {
    "pos": 0.5,  # Positive sentiment is given higher weight
    "neg": 0.3,  # Negative sentiment has medium weight
    "neu": 0.2   # Neutral sentiment has lower weight
}

# Example prompt
prompt = "I love programming, but it can sometimes be frustrating."

# Get the best model based on weighted sentiment scores
result = get_weighted_sentiment_model(prompt, llms, weights)
print(f"The best model is {result['llm']} with a weighted score of {result['weighted_score']:.4f}")

