import heapq
import logging
import time
import math
import numpy as np
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity
from indexer import clean_text, preprocess_text, get_model, spacy_preprocess
from rank_bm25 import BM25Okapi
from functools import lru_cache
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

# Setup logger
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Load sentence transformer model
model = get_model()

@lru_cache(maxsize=10000)
def encode_cached(text: str):
    return model.encode(text)

# URL normalization for deduplication
def normalize_url(url):
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or '/'
    if path.endswith("/index.html"):
        path = path[:-11] or '/'
    if path != '/':
        path = path.rstrip("/")
    query = urlencode(sorted(parse_qsl(parsed.query)))
    return urlunparse((scheme, netloc, path, '', query, ''))

def expand_query(query, model, threshold=0.6, max_synonyms=3):
    tokens = spacy_preprocess(query)
    expanded = set(tokens)
    q_vec = model.encode(query)

    for token in tokens:
        synsets = wn.synsets(token)
        for syn in synsets[:1]:
            for lemma in syn.lemmas()[:max_synonyms]:
                synonym = lemma.name().replace('_', ' ')
                if synonym != token and len(synonym.split()) <= 2:
                    sim = cosine_similarity([q_vec], [encode_cached(synonym)])[0][0]
                    if sim >= threshold and all(len(w) > 2 and w.isalpha() for w in synonym.split()):
                        expanded.add(synonym)
    return list(expanded)

def semantic_score(query, doc_embeddings, model):
    q_vec = model.encode(query)
    return {
        url: cosine_similarity([q_vec], [d_vec])[0][0]
        for url, d_vec in doc_embeddings.items()
    }

def build_bm25_index(docs):
    urls = list(docs.keys())
    cleaned_docs = [" ".join(preprocess_text(docs[url])) for url in urls]
    tokenized_corpus = [doc.split() for doc in cleaned_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, urls

def normalize_scores(scores):
    vals = list(scores.values())
    min_val, max_val = min(vals), max(vals)
    if max_val == min_val:
        return {k: 1 for k in scores}
    return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}


def search(query, pagerank, embeddings, bm25, bm25_urls, K=100):
    start_time = time.time()
    query_len = len(query.split())
    logging.info(f"[SEARCH] Query: '{query}' | Length: {query_len} | K: {K}")
 
    if query_len <= 3:
        alpha, beta, gamma = 0.6, 0.3, 0.1
    elif query_len >= 10:
        alpha, beta, gamma = 0.4, 0.6, 0.2
    else:
        alpha, beta, gamma = 0.5, 0.5, 0.15

    query_clean = clean_text(query)
    if not query_clean:
        logging.warning(f"Query cleaned to empty: '{query}' → '{query_clean}'")
        return []

    expanded_query = expand_query(query, model)
    logging.info(f"[EXPANSION] '{query}' → {expanded_query}")
    query_tokens = preprocess_text(" ".join(expanded_query))
    if not query_tokens:
        logging.warning(f"Query tokens empty after preprocessing: '{query}'")
        return []

    # --- BM25 Scoring ---
    bm25_start = time.time()
    bm25_scores_raw = dict(zip(bm25_urls, bm25.get_scores(query_tokens)))
    top_k_urls = [(u, s) for u, s in bm25_scores_raw.items() if s > 0.01]
    top_k_urls = heapq.nlargest(min(len(top_k_urls), 50), top_k_urls, key=lambda x: x[1])
    candidate_urls = [url for url, _ in top_k_urls]
    logging.info(f"[BM25] {len(top_k_urls)} top candidates selected. ⏱ {time.time() - bm25_start:.2f}s")

    if not candidate_urls:
        logging.warning("[BM25] No matches. Falling back to semantic similarity.")
        try:
            q_embed = model.encode(query).reshape(1, -1)
            semantic_raw_scores = semantic_score(query, embeddings, model)
            
            top_semantic = heapq.nlargest(30, semantic_raw_scores.items(), key=lambda x: x[1])
            candidate_urls = [url for url, _ in top_semantic]
        except Exception as e:
            logging.error(f"[SEMANTIC] Fallback failed: {e}")
            candidate_urls = sorted(pagerank, key=pagerank.get, reverse=True)[:30]

    # --- Semantic Scoring ---
    semantic_start = time.time()
    q_embed = model.encode(spacy_preprocess(query), convert_to_numpy=True)
    doc_vecs = [(url, embeddings[url]) for url in candidate_urls if url in embeddings]
    if not doc_vecs:
        logging.warning("[SEMANTIC] No embeddings found for candidate URLs.")
        return []

    urls, vecs = zip(*doc_vecs)
    sim_array = cosine_similarity(q_embed, np.array(vecs)).flatten()
    sim_scores = dict(zip(urls, sim_array))
    logging.info(f"[SEMANTIC] Scored {len(sim_scores)} docs. ⏱ {time.time() - semantic_start:.2f}s")

    # --- PageRank Scoring ---
    pr_scores = {url: pagerank.get(url, 0) for url in candidate_urls}
    logging.info(f"[PAGERANK] Applied to {len(pr_scores)} URLs.")

    # --- Normalize ---
    bm25_topk = {url: bm25_scores_raw[url] for url in candidate_urls}
    bm25_scores = normalize_scores(bm25_topk)
    sim_scores = normalize_scores(sim_scores)
    pr_scores = normalize_scores(pr_scores)

    # --- Final Aggregation ---
    final_scores = {}
    for url in set(bm25_scores) | set(sim_scores) | set(pr_scores):
        final_scores[url] = (
            alpha * bm25_scores.get(url, 0) +
            beta * sim_scores.get(url, 0) +
            gamma * pr_scores.get(url, 0)
        )

    ranked = heapq.nlargest(K, final_scores.items(), key=lambda x: x[1])

    # --- Deduplicate Final Results by Normalized URL ---
    seen = set()
    res = []
    for url, score in ranked:
        norm = normalize_url(url)
        if norm not in seen:
            seen.add(norm)
            res.append((url, score))
        if len(res) >= 10:
            break

    logging.info(f"[RESULT] Top {len(res)} results computed. ⏱ Total: {time.time() - start_time:.2f}s")

    for url, score in res[:5]:
        logging.info(f"-> {url[:60]}... | Final: {score:.4f} | "
                     f"BM25: {bm25_scores.get(url, 0):.4f}, "
                     f"Sim: {sim_scores.get(url, 0):.4f}, PR: {pr_scores.get(url, 0):.4f}")

    return res
