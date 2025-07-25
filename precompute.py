# precompute.py
import json
import pickle
from indexer import build_bm25_index,get_model
from pageranker import build_pagerank
from indexer import get_model
model = get_model()

def load_data(file='url_content.txt'):
    docs, titles, snippets = {}, {}, {}
    with open(file, encoding='utf-8', errors='ignore') as f:
        block = {}
        for line in f:
           
            if line.startswith('@@URL@@'):
                block['url'] = line[8:].strip()
                if block.get('url') in docs:
                    continue

            elif line.startswith('@@TITLE@@'):
                title = line[10:].strip()
                if title.lower() in ['home', 'index', 'faq', 'about']:
                    continue
                block['title'] = title
                if not block.get('title'):
                    block['title'] = block['url']


            elif line.startswith('@@CONTENT@@'):
                block['content'] = line[12:].strip()
            elif line.startswith('@@END@@'):
                if block.get('url') and block.get('content'):
                    content = block['content']
                    if block.get('url') in docs or not block.get('content') or len(block['content'].split()) < 40:
                        continue

                    u = block['url']
                    docs[u] = content
                    titles[u] = block.get('title', 'No Title')
                    snippets[u] = content[:300]
                block = {}
    return docs, titles, snippets


def load_graph(file='url_graph.json'):
    with open(file) as f:
        data = json.load(f)
    edges = []
    if isinstance(data, dict):
        for u, vs in data.items():
            for v in vs:
                edges.append((u, v))
    elif isinstance(data, list):
        edges = data
    else:
        raise ValueError("Invalid graph format")
    return build_pagerank(edges)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

if __name__ == '__main__':
    print("[INFO] Loading data...")
    docs, titles, snippets = load_data()


    print("[INFO] Building BM25 index...")
    bm25, bm25_urls = build_bm25_index(docs)

    print("[INFO] Building PageRank...")
    pagerank = load_graph()

    print("[INFO] Embedding documents...")
    model = get_model()
    all_texts = list(docs.values())
    embeddings = model.encode(all_texts, show_progress_bar=True, batch_size=32)
    print("[INFO] Embeddings computed.")
    url_to_embedding = dict(zip(docs.keys(), embeddings))

    # Save all outputs
    save_pickle((bm25, bm25_urls), 'bm25_index.pkl')
    save_pickle(url_to_embedding, 'embeddings.pkl')
    save_pickle(pagerank, 'pagerank.pkl')
    save_pickle((docs, titles, snippets), 'content.pkl')


    print("Precomputation complete!") 
    print(f"Documents indexed (BM25): {len(bm25_urls)}")
    print(f"Documents embedded: {len(url_to_embedding)}")
    print(f"PageRank nodes: {len(pagerank)}")


