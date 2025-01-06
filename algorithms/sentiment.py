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
