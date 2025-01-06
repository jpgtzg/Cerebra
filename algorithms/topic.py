# Written by Juan Pablo Gutiérrez
# 06/01/2025
# This file handles topic based selection of a LLM model.

from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import nltk

from algorithms.utils import similarity_score

vectorizer = CountVectorizer()

def get_topic_based_model(model: SentenceTransformer, prompt: str, llms: dict) -> str:
    """
    Gets the best topic based model for the given prompt.

    Args:
        model (SentenceTransformer): The semantic model to use.
        prompt (str): The prompt to analyze.
        llms (dict): The LLM models to decide on.

    Returns:
        str: The best topic based model.
    """
    tokens = nltk.word_tokenize(prompt.lower())

    X = vectorizer.fit_transform([' '.join(tokens)])

    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    topic_distributions = lda.fit_transform(X)

    topic_based_models = {}

    # Main part: Loop through each descriptions to calculate similarities with the LLM models
    for topic_idx, topic_distribution in enumerate(lda.components_):
        topic_desc = generate_topic_description(topic_distribution)
        # Calculate similarity scores for each LLM model
        similarities = {model_name: similarity_score(model, topic_desc, llms[model_name]) for model_name in llms.keys()}
        best_match = max(similarities, key=similarities.get)
        topic_based_models[topic_idx] = {
            "llm": best_match,
            "similarity": similarities[best_match]
        }

    best_topic = np.argmax(topic_distributions)
    result = topic_based_models[best_topic]

    return {
        "best_match": result['llm'],
        "similarity": result['similarity']
    }

def generate_topic_description(topic_distribution, top_n_words=10):
    """
    Extracts the topic description from the topic distribution.
    """
    words = vectorizer.get_feature_names_out()
    top_words_idx = np.argsort(topic_distribution)[-top_n_words:]
    top_words = [words[i] for i in top_words_idx]
    return ' '.join(top_words)

