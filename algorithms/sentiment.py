# Written by Juan Pablo Gutiérrez
# 06/01/2025
# This file handles sentiment analysis of a prompt.

from nltk.sentiment import SentimentIntensityAnalyzer

def get_sentiment_analysis(prompt: str) -> str:
    """
    Gets the sentiment analysis of the given prompt.

    Args:
        prompt (str): The prompt to analyze.

    Returns:
        str: The sentiment analysis of the prompt.
    """
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(prompt)

def get_sentiment_based_model(prompt: str, llms: dict) -> str:
    """
    Gets the best sentiment based model for the given prompt.

    Args:
        prompt (str): The prompt to analyze.
        llms (dict): The LLM models to decide on.

    Returns:
        str: The best sentiment based model for the given prompt.
    """
    return llms[get_sentiment_analysis(prompt)]

llms = {
    "positive": "gpt-4o",
    "negative": "gpt-4o",
    "neutral": "gpt-4o"
}

prompt = "I love programming"
res = get_sentiment_analysis(prompt)
print(res)