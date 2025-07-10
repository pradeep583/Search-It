from collections import defaultdict

def build_pagerank(edges, damping=0.85, iterations=20):
    outlinks = defaultdict(set)
    inlinks = defaultdict(set)
    urls = set()

    # Build graph
    for u, v in edges:
        outlinks[u].add(v)
        inlinks[v].add(u)
        urls.update([u, v])

    if not urls:
        return {}

    N = len(urls)
    rank = {u: 1 / N for u in urls}

    for _ in range(iterations):
        new_rank = {}
        # Identify dangling nodes: no outlinks
        dangling_nodes = [u for u in urls if len(outlinks[u]) == 0]
        dangling_mass = sum(rank[u] for u in dangling_nodes)

        for u in urls:
            # Start with teleportation component
            new_rank[u] = (1 - damping) / N
            # Add dangling node contribution
            new_rank[u] += damping * dangling_mass / N
            # Add actual incoming PageRank from inlinks
            incoming_score = sum(
                damping * rank[v] / len(outlinks[v])
                for v in inlinks[u]
                if len(outlinks[v]) > 0
            )
            new_rank[u] += incoming_score

        # Normalize
        # Final scaling: normalize to max 1.0
        max_pr = max(rank.values()) + 1e-9  # Avoid divide by zero
        rank = {u: r / max_pr for u, r in rank.items()}

        
    #logging
    print(f"[INFO] PageRank computed with {len(urls)} URLs and {len(edges)} edges.")

    return rank