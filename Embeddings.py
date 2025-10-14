def load_docs():
    import pandas as pd
    df = pd.read_csv(WORKS_CSV)
    docs = []
    for _, r in df.iterrows():
        text = f"{r.get('title','')}\n\n{r.get('abstract','')}"
        meta = {k: str(r.get(k,"")) for k in ["pubmed_id","journal","year","category","title"]}
        docs.append(Document(page_content=text, metadata=meta))
    return docs

def cmd_index():
    ensure_dirs()
    info("Loading works.csv …")
    docs = load_docs()
    info(f"{len(docs)} docs loaded.")

    info(f"Embedding with Ollama model: {EMBED_MODEL}")
    emb = OllamaEmbeddings(model=EMBED_MODEL)

    info(f"Building Chroma at: {CHROMA_DIR}")
    vectordb = Chroma.from_documents(docs, emb, persist_directory=CHROMA_DIR)
    vectordb.persist()
    ok("Index built.")