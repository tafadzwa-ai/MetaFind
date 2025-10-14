
def recommend_vector(query: str):
    emb = OllamaEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb)
    results = vectordb.similarity_search_with_score(query, k=TOP_K)

    buckets = collections.defaultdict(list)
    for doc, distance in results:
        # Convert distance → similarity-like score
        sim = 1.0 / (1.0 + float(distance))
        buckets[doc.metadata.get("journal","Unknown")].append((sim, doc))

    ranked = []
    for venue, lst in buckets.items():
        sims = [s for s,_ in lst]
        mean_s, max_s, n = sum(sims)/len(sims), max(sims), len(sims)
        score = 0.7*mean_s + 0.3*max_s + 0.05*math.log(1+n)
        evid = sorted(lst, key=lambda x:x[0], reverse=True)[:EVIDENCE_K]
        ranked.append((score, venue, evid))
    ranked.sort(key=lambda x:x[0], reverse=True)
    return ranked[:TOP_VENUES]

def cmd_recommend(query: str):
    top = recommend_vector(query)
    print("\n=== Vector Index Recommender ===")
    print(f"Query: {query}\n")
    for i,(score, venue, evid) in enumerate(top, 1):
        print(f"{i}) {venue} — score={score:.3f}")
        for sim, doc in evid:
            m = doc.metadata
            print(f"   - {m.get('title','')[:80]} … ({m.get('year','')}) [sim≈{sim:.3f}] PMID:{m.get('pubmed_id','')}")
    print()