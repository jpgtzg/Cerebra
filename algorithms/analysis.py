# Written by Juan Pablo Gutiérrez
# 05/01/2025
# This file handles analysis of a prompt.

from sentence_transformers import SentenceTransformer, util

llms = {
    "gpt-4": "Complex tasks, reasoning, creativity, and highly complex tasks. Advanced reasoning and creativity. Excels at complex, multi-step tasks",
    "gpt-4o": "Moderately optimized for reasoning and cost-efficiency. Balanced for moderate tasks		",
    "gpt-3.5": "Simple tasks, reasoning, creativity, and mid-level complexity tasks. Solid reasoning but less depth. Handles mid-level tasks"
}

prompt = "How does climate change affect biodiversity?"

model = SentenceTransformer('all-MiniLM-L12-v2')

def semantic_similarity(prompt: str, semantic_model: str, llm_model: str) -> float:
    prompt_embedding = semantic_model.encode(prompt)
    description_embedding = semantic_model.encode(llms[llm_model])
    return util.pytorch_cos_sim(prompt_embedding, description_embedding)

def get_best_match(prompt: str, semantic_model: str) -> str:
    similarities = {model_name: semantic_similarity(prompt, semantic_model, model_name) for model_name in llms.keys()}
    best_match = max(similarities, key=similarities.get)
    return {
        "best_match": best_match,
        "similarity": similarities[best_match]
    }

res = get_best_match(prompt, model)
print(f"The best match is {res['best_match']} with a similarity of {res['similarity']}")

# Topic based selection

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import nltk

vectorizer = CountVectorizer()
tokens = nltk.word_tokenize(prompt.lower())

X = vectorizer.fit_transform([' '.join(tokens)])

lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda_topics = lda.fit_transform(X)

def generate_topic_description(topic_distribution, top_n_words=10):
    # Extract top words for each topic
    words = vectorizer.get_feature_names_out()
    top_words_idx = np.argsort(topic_distribution)[-top_n_words:]
    top_words = [words[i] for i in top_words_idx]
    return ' '.join(top_words)

def similarity_score(desc1, desc2):
    # Compute similarity between two descriptions
    embedding1 = model.encode(desc1)
    embedding2 = model.encode(desc2)
    return util.pytorch_cos_sim(embedding1, embedding2).item()

# Mapping topics to LLMs based on similarity
topic_based_models = {}

for topic_idx, topic_distribution in enumerate(lda.components_):
    topic_desc = generate_topic_description(topic_distribution)
    print(topic_desc)
    similarities = {model_name: similarity_score(topic_desc, llms[model_name]) for model_name in llms.keys()}
    best_match = max(similarities, key=similarities.get)
    topic_based_models[topic_idx] = {
        "llm": best_match,
        "similarity": similarities[best_match]
    }

# Selecting the best match for the given prompt
best_topic = np.argmax(lda_topics)
result = topic_based_models[best_topic]
print(f"The best match is {result['llm']} with a similarity of {result['similarity']}")