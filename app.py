# app.py
from flask import Flask, request, render_template
import pickle
from searcher import search


app = Flask(__name__)

# Load everything from precomputed files
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

print("[INFO] Loading precomputed files...")
bm25, bm25_urls = load_pickle('bm25_index.pkl')
pagerank = load_pickle('pagerank.pkl')
embeddings = load_pickle('embeddings.pkl')
docs, titles, snippets = load_pickle('content.pkl')
print("[ READY] App booted instantly with precomputed data.")

@app.route("/", methods=["GET", "POST"])
def home():
    q = ""
    results = []
    if request.method == "POST":
        q = request.form['query']
        if q.strip():
            try:
                raw = search(q, pagerank, embeddings, bm25, bm25_urls) 
                for url, scores in raw:
                    results.append({
                        'title': titles.get(url, 'No Title'),
                        'url': url,
                        'snippet': snippets.get(url, 'No snippet available')
                    })
            except Exception as e:
                print(f"[ERROR] Search failed: {e}")
    return render_template("index.html", results=results ,query=q)


if __name__ == "__main__":
    app.run(debug=True)

