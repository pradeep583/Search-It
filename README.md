# Search Engine

A lightweight yet powerful web search engine combining:

- **BM25** for keyword-based relevance  
- **BERT embeddings** for semantic understanding  
- **PageRank** for link-based importance

## Goals

- Go beyond keyword matching by understanding intent
- Rank results by both *relevance* and authority
- Deliver fast performance with offline computation  
- Serve as a solid foundation for building more advanced IR systems  


## Requirements

- Python 3.8+
- Flask – lightweight web framework  
- `sentence-transformers` – for BERT-based embeddings  
- `rank-bm25` – for BM25 scoring  
- `spaCy` (with `en_core_web_sm`) – for NLP preprocessing  
- `beautifulsoup4` – to clean HTML content  
- `pickle`, `numpy`, `scikit-learn` – utils & storage

### Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```


## To Get Started

### 1. Prepare your data

```bash
python crawler.py
```
* This will crawl static webpages and store them in url_content.txt and make nodes in url_graph.json which would be like
  
* url_content.txt in this format:

  @@URL@@ https://example.com
  @@TITLE@@ Example Title
  @@CONTENT@@ Actual page content...
  @@END@@
  
* graph (links between URLs) in url_graph.json:

  json
  {
    "https://example.com": ["https://example.com/about", "https://example.com/contact"],
    ...
  }
  

### 2. Run precomputation

bash
python precompute.py

*  This will help to precompute all the crawled data in the pickle files which can be later used for retrieving soon.

### 3. Run app

bash
python app.py

* This will run the app and bring the data from precomputed pickle files.

## Tech Stack

* *NLP*: SpaCy, BERT (MiniLM)
* *IR*: BM25 (Okapi), Sentence Embeddings
* *Graph Theory*: PageRank Algorithm
* *Web*: Flask + HTML

## Contribution 
 * Pull requests and feature suggestions are welcome!
 * Feel free to fork, test, and enhance this project.

## Contact
 - 📧 Email: [pradeepravikumar1@gmail.com](mailto:pradeepravikumar1@gmail.com)
 - 💬 DM me directly on [LinkedIn](https://www.linkedin.com/in/pradeep-ravikumar)


