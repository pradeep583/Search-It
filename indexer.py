# indexer.py
import nltk
nltk.download('wordnet') # For first time 

import re
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import torch

# Load SpaCy & Model
nlp = spacy.load("en_core_web_sm")
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model():
    global model
    if model is None:
        print("[INFO] Loading SentenceTransformer...")
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"[ERROR] Failed to load SentenceTransformer: {e}")
            raise
        model.eval()
    return model

# Cleaners 
def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'noscript']):
        tag.decompose()
    return soup.get_text(separator=' ', strip=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tokenizers
def preprocess_text(text):
    if not text:
        return []
    text = re.sub(r'[^a-z\s]', '', text.lower())

    tokens = text.split()
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 2]


def spacy_preprocess(text):
    doc = nlp(text.lower())
    return [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop 
    ]

# BM25
def build_bm25_index(docs):
    preprocessed = {}
    for url, content in docs.items():
        tokens = preprocess_text(content)
        if len(tokens) >= 5:
            preprocessed[url] = tokens
    bm25 = BM25Okapi(list(preprocessed.values()))
    return bm25, list(preprocessed.keys())

# Sentence-BERT
def batched_encode(texts, batch_size=64):
    model = get_model()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Batches"):
            batch = texts[i:i + batch_size]
            embs = model.encode(batch, convert_to_numpy=True)
            embeddings.extend(embs)
    return np.array(embeddings)

def build_bert_embeddings(docs):
    processed = {url: " ".join(spacy_preprocess(content)) for url, content in docs.items()}
    urls, texts = list(processed.keys()), list(processed.values())
    embeddings = batched_encode(texts)
    return dict(zip(urls, embeddings))
