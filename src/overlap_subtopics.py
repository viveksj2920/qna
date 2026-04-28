import pandas as pd
import numpy as np
from azure_search.index_handler import AzureSearchIndexUtility
from skillsets.vectorizer import AzureOpenAIVectorizer
from tqdm import tqdm
from rapidfuzz import fuzz
import re

from dotenv import load_dotenv
load_dotenv()

import time
import random


def embed_with_retry(client, model, batch, max_retries=10):
    for i in range(max_retries):
        try:
            return client.embeddings.create(input=batch, model=model)
        except Exception as e:
            if "RateLimitReached" in str(e) or "429" in str(e):
                wait = (2 ** i) + random.uniform(0, 0.5)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Too many 429 rate limit errors; try again later.")


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def compute_lexical_similarity(text1, text2):
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    ratio = fuzz.ratio(norm1, norm2) / 100.0
    token_sort = fuzz.token_sort_ratio(norm1, norm2) / 100.0
    return max(ratio, token_sort)


def compute_semantic_similarity(emb1, emb2):
    return np.dot(emb1, emb2)


# only skip lexical for tiny single-word tokens (a, e, to, an, etc.)
def has_tiny_word_only(s, min_len=3):
    tokens = re.findall(r"[A-Za-z0-9]+", s.lower())
    return len(tokens) == 1 and len(tokens[0]) < min_len


def main():
    index_handler = AzureSearchIndexUtility(index_name="transcripts-qna-subtopic-groupings")
    results = index_handler.search(search_text="*", select=["grouped_subtopic"])
    df = pd.DataFrame(results)

    grouped_subtopics = (
        df["grouped_subtopic"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    grouped_subtopics = grouped_subtopics[grouped_subtopics != ""].unique().tolist()

    print(f"Found {len(grouped_subtopics)} unique grouped subtopics")

    print(f"Generating embeddings for {len(grouped_subtopics)} grouped subtopics...")
    vectorizer = AzureOpenAIVectorizer()
    embeddings = []
    BATCH_SIZE = 16  # reduced to avoid 429 rate limit

    for i in tqdm(range(0, len(grouped_subtopics), BATCH_SIZE), desc="Generating embeddings"):
        batch = grouped_subtopics[i:i + BATCH_SIZE]
        response = embed_with_retry(vectorizer.client, vectorizer.model, batch)
        batch_vectors = [item.embedding for item in response.data]
        embeddings.extend(batch_vectors)
        time.sleep(0.2)  # small throttle between batches

    embeddings = np.array(embeddings)
    embeddings_normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    print(f"\nGenerated embeddings with shape: {embeddings.shape}")

    print("Computing lexical similarity matrix...")
    n = len(grouped_subtopics)
    lexical_matrix = np.zeros((n, n), dtype=np.float32)

    for i in tqdm(range(n), desc="Computing lexical similarities"):
        for j in range(i, n):
            score = compute_lexical_similarity(grouped_subtopics[i], grouped_subtopics[j])
            lexical_matrix[i, j] = score
            lexical_matrix[j, i] = score

    print(f"Shape: {lexical_matrix.shape}")

    print("Computing semantic similarity matrix...")
    semantic_matrix = embeddings_normalized @ embeddings_normalized.T
    print(f"Shape: {semantic_matrix.shape}")

    LEXICAL_THRESHOLD = 0.85
    SEMANTIC_THRESHOLD = 0.85

    overlap_results = []
    for i in range(len(grouped_subtopics)):
        for j in range(i + 1, len(grouped_subtopics)):
            topic_1 = grouped_subtopics[i]
            topic_2 = grouped_subtopics[j]

            lexical_similarity = lexical_matrix[i, j]
            semantic_similarity = semantic_matrix[i, j]

            # skip lexical only for tiny single-word tokens
            if has_tiny_word_only(topic_1) or has_tiny_word_only(topic_2):
                lexical_similarity = 0.0

            if (lexical_similarity >= LEXICAL_THRESHOLD) or (semantic_similarity >= SEMANTIC_THRESHOLD):
                criteria_met = []
                if lexical_similarity >= LEXICAL_THRESHOLD:
                    criteria_met.append("Lexical")
                if semantic_similarity >= SEMANTIC_THRESHOLD:
                    criteria_met.append("Semantic")

                overlap_results.append({
                    "topic_1": topic_1,
                    "topic_2": topic_2,
                    "lexical_score": lexical_similarity,
                    "semantic_score": semantic_similarity,
                    "max_score": max(lexical_similarity, semantic_similarity),
                    "detected_by": " + ".join(criteria_met)
                })

    print(f"{len(overlap_results)} overlapping pairs")

    overlaps_df = pd.DataFrame(overlap_results)
    if len(overlaps_df) > 0:
        overlaps_df = overlaps_df.sort_values("max_score", ascending=False)
        print(f"Total topics: {len(grouped_subtopics)}")
        print(f"Total overlapping pairs: {len(overlaps_df)}")
        print(overlaps_df[["topic_1", "topic_2", "lexical_score", "semantic_score", "detected_by"]].head(30))
    else:
        print("No overlapping pairs")

    output_path = "overlap_first_100.csv"
    overlaps_df.to_csv(output_path, index=False)  # save correct dataframe
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()