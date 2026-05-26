from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd
import fitz

#load model
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.load("embeddings.npy")
papers = pd.read_json("metadata.json")

#similar paper retrieval 
def retrieve_similar_papers(query: str, top_n:int = 5):
    #generate embeddings
    query_embeddings = model.encode(query)
    #compute similarity
    similarities = cosine_similarity([query_embeddings],embeddings)[0]

    #get top matches
    top_indices = similarities.argsort()[-top_n:][::-1]
    results = []

    for idx in top_indices:
        paper = papers.iloc[idx]
        results.append({
            "title":paper["title"],
            "similarity":float(similarities[idx]),
        })
    return results
