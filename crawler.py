# === DFS-Based Multi-threaded Web Crawler with Resume Checkpoints (Fixed) ===
import os
import time
import json
import string
import logging
import threading
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
from urllib.parse import urlparse, urljoin, urlunparse
from urllib.robotparser import RobotFileParser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor

# === CONFIGURATION ===
CRAWL_DELAY = 0.3
MAX_PAGES_PER_DOMAIN = 15
MAX_PAGES_TOTAL = 30
VISITED_FILE = 'visited.txt'
GRAPH_FILE = 'url_graph.json'
OUTPUT_FILE = 'url_content.txt'
CHECKPOINT_DIR = 'checkpoints'
THREAD_COUNT = 8
MAX_DEPTH = 5

EXCLUDE_PATTERNS = [
    "facebook", "login", "signup", "donate", "mailto:", "tel:",
    "privacy", "terms", "about", "contact", "help", "support",
    "itunes.apple.com", "apps.apple.com", "play.google.com", "appstore",
    "apkcombo", "twitter"
]
IGNORE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.pdf',
                     '.zip', '.mp4', '.webp', '.svg', '.woff2')

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/114.0.0.0 Safari/537.36"
}

# === LOGGING ===
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO
)

# === THREAD GLOBALS ===
thread_local = threading.local()
file_lock = threading.Lock()
graph_lock = threading.Lock()
visited_lock = threading.Lock()
crawl_count_lock = threading.Lock()
edges_global = []
domain_last_access = defaultdict(float)
global_crawl_count = 0

def get_session():
    if not hasattr(thread_local, "session"):
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        session.headers.update(HEADERS)
        thread_local.session = session
    return thread_local.session

def join_and_normalize(base, href):
    joined = urljoin(base, href)
    return normalize_url(joined)

def normalize_url(url):
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip('/') or '/'
    return urlunparse((scheme, netloc, path, '', '', ''))

def is_valid_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        return False
    return not any(url.lower().endswith(ext) for ext in IGNORE_EXTENSIONS)

def is_relevant(url):
    return all(p not in url.lower() for p in EXCLUDE_PATTERNS)

def is_mostly_english(text, threshold=0.7):
    total = len(text)
    ascii_chars = sum(1 for c in text if c in string.printable)
    return (ascii_chars / total) >= threshold if total else False

def throttle(domain):
    if not hasattr(thread_local, "domain_access"):
        thread_local.domain_access = defaultdict(float)

    elapsed = time.time() - thread_local.domain_access[domain]
    if elapsed < CRAWL_DELAY:
        time.sleep(CRAWL_DELAY - elapsed)
    thread_local.domain_access[domain] = time.time()

def get_checkpoint_path(seed_url):
    domain = urlparse(seed_url).netloc.replace(":", "_")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"{domain}_stack.json")

def save_stack_checkpoint(seed_url, stack):
    with file_lock:
        with open(get_checkpoint_path(seed_url), 'w', encoding='utf-8') as f:
            json.dump(stack, f)

def load_stack_checkpoint(seed_url):
    path = get_checkpoint_path(seed_url)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except:
                logging.warning(f"Invalid checkpoint format for {seed_url}")
    return []

def is_allowed(url, rp_map):
    domain = urlparse(url).netloc
    if domain not in rp_map:
        rp = RobotFileParser()
        rp.set_url(f"https://{domain}/robots.txt")
        try:
            rp.read()
            rp_map[domain] = rp
        except:
            rp_map[domain] = None
    rp = rp_map[domain]
    return rp is None or rp.can_fetch("*", url)

def scrape(url):
    try:
        session = get_session()
        r = session.get(url, timeout=5, allow_redirects=True)
        if r.status_code != 200:
            return None, None, None

        soup = BeautifulSoup(r.text, 'html.parser')
        [tag.decompose() for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'form'])]

        text = ' '.join(t.get_text(strip=True) for t in soup.find_all(['p', 'li', 'h1', 'h2', 'h3']))
        title = soup.title.get_text(strip=True) if soup.title else ""
        return soup, title, text
    except Exception as e:
        logging.warning(f"[SCRAPE ERROR] {url} - {e}")
        return None, None, None

def load_visited():
    if not os.path.exists(VISITED_FILE):
        return set()
    with open(VISITED_FILE, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)

def save_visited_batch(batch):
    with file_lock:
        with open(VISITED_FILE, 'a', encoding='utf-8') as f:
            for url in batch:
                f.write(url + '\n')

def bfs_crawl(seed, visited, rp_map):
    global global_crawl_count

    local_stack = load_stack_checkpoint(seed) or [(seed, 0)]
    local_visited = set()
    local_edges = []
    domain_hits = defaultdict(int)

    with open(OUTPUT_FILE, 'a', encoding='utf-8', errors='replace') as out_file:
        while local_stack:
            url, depth = local_stack.pop()
            url = normalize_url(url)

            if depth > MAX_DEPTH or len(local_visited) >= MAX_PAGES_TOTAL:
                continue

            domain = urlparse(url).netloc

            with visited_lock:
                if url in visited or domain_hits[domain] >= MAX_PAGES_PER_DOMAIN:
                    continue
                visited.add(url)
                local_visited.add(url)

            if not is_allowed(url, rp_map):
                logging.info(f"[BLOCKED BY ROBOTS] {url}")
                continue

            throttle(domain)
            soup, title, content = scrape(url)

            if not content:
                logging.info(f"[SKIP] Empty content: {url}")
                continue
            if len(content.split()) < 50:
                logging.info(f"[SKIP] Too short: {url}")
                continue
            if not is_mostly_english(content):
                logging.info(f"[SKIP] Not English: {url}")
                continue


            with file_lock:
                out_file.write(f"@@URL@@ {url}\n")
                out_file.write(f"@@TITLE@@ {title}\n")
                out_file.write(f"@@CONTENT@@ {content}\n")
                out_file.write("@@END@@\n\n")

            with crawl_count_lock:
                global_crawl_count += 1
                log_count = global_crawl_count
            logging.info(f"[{log_count}] Crawled: {url}")

            domain_hits[domain] += 1

            for a in soup.find_all('a', href=True):
                next_url = join_and_normalize(url, a['href'])
                if is_valid_url(next_url) and is_relevant(next_url):
                    with visited_lock:
                        if next_url not in visited:
                            local_stack.append((next_url, depth + 1))
                            local_edges.append((url, next_url))

            save_stack_checkpoint(seed, local_stack)

    save_visited_batch(local_visited)
    with graph_lock:
        edges_global.extend(local_edges)

def crawl_and_save(seeds):
    rp_map = {}
    visited = load_visited()

    with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        futures = [executor.submit(bfs_crawl, seed, visited, rp_map) for seed in seeds]
        for future in futures:
            future.result()

    existing_edges = []
    if os.path.exists(GRAPH_FILE):
        with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
            try:
                existing_edges = json.load(f)
            except:
                logging.warning("Invalid graph file format")

    combined = list({(src, dst) for src, dst in (existing_edges + edges_global)})
    with open(GRAPH_FILE, 'w', encoding='utf-8') as f:
        json.dump([list(edge) for edge in combined], f, indent=2)

    logging.info(f"*Crawl finished. Total unique edges: {len(combined)}*")

if __name__ == "__main__":
    seed_urls = [
        "https://www.carwale.com/new/best-cars/",
        "https://www.bikewale.com/best-bikes-in-india/"



    ]
    crawl_and_save(seed_urls)
