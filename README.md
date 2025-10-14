# MetaFind: An AI-Enhanced Venue Recommendation System

MetaFind is a Streamlit web app that helps academic researchers discover suitable journals for their manuscripts via two complementary methods:

1) **Data-driven vector search** over a **local PubMed-derived** database (ChromaDB + `nomic-embed-text` embeddings).
2) **LLM-assisted web search** that uses a **local Ollama LLM (`gemma:2b`)** to brainstorm relevant venues/articles, then calls the **Tavily** web search API to return real, clickable links.

> **Why two paths?** The vector search provides grounded, fast retrieval from a vetted local corpus. The LLM-assisted path adds breadth and recency by ideating and verifying with live web results.

---

## Features

- **Journal Recommendation (Vector Search):**
  - Local ChromaDB index built from PubMed data.
  - Embedding model: `nomic-embed-text` (via Ollama) for semantic similarity.
  - Fast, offline-capable querying after one-time indexing.

- **LLM-Assisted Web Search:**
  - Local LLM (`gemma:2b` via Ollama) to brainstorm candidate journals & query angles.
  - **Tavily** tool integration to fetch reliable, clickable links that validate LLM ideas.

- **Streamlit UI:**
  - Single-page app with clear inputs (title/abstract/keywords).
  - Side-by-side results from both methods for triangulation.

- **Local-first Design:**
  - All LLM inference runs locally via Ollama.
  - Only Tavily calls leave your machine (for web results).

---

## Project Structure

```
.
├─ app.py               # Streamlit frontend (UI + orchestration)
├─ MetaFinder2.py       # Backend logic (data build, vector search, LLM-assisted web search)
├─ requirements.txt     # Python dependencies
├─ README.md            # This file
├─ .gitignore           # Git ignore rules
├─ journal_data/        # (generated) parsed/cleaned PubMed-derived data
└─ journal_chroma_db/   # (generated) ChromaDB persistent store
```

**Main Files**
- **`app.py`** — Streamlit app that exposes both recommendation paths and renders results.
- **`MetaFinder2.py`** — Library/CLI for:
  - **One-time data build** (ingest + chunk + embed + persist to ChromaDB).
  - **Query helpers** for vector search and LLM-assisted Tavily search.

---

## How to Run Locally

### Prerequisites

1. **Python 3.10+** recommended.
2. **Git** installed.
3. **Ollama** installed and running locally  
   - Install: https://ollama.com/download  
   - Pull models used by MetaFind:
     ```bash
     ollama pull nomic-embed-text
     ollama pull gemma:2b
     ```
4. **Tavily API key** (free/paid tiers available)  
   - Create an account and get an API key: https://app.tavily.com/  
   - You will export it as `TAVILY_API_KEY` below.

> **Note:** Ollama must be running (usually at `http://localhost:11434`) before building the index or querying.

---

### Setup Instructions

#### 1) Clone the repository
```bash
git clone https://github.com/<your-username>/MetaFind.git
cd MetaFind
```

#### 2) Create and activate a virtual environment
```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### 3) Install Python packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4) Add your Tavily API key to the environment
```bash
# macOS / Linux (temporary for current shell)
export TAVILY_API_KEY="YOUR_TAVILY_API_KEY"

# macOS / Linux (persist in .env for your shell profile if desired)
echo 'export TAVILY_API_KEY="YOUR_TAVILY_API_KEY"' >> ~/.bashrc    # or ~/.zshrc

# Windows (PowerShell)
setx TAVILY_API_KEY "YOUR_TAVILY_API_KEY"
```

#### 5) One-time data setup (build local vector store)
This step ingests your PubMed-derived dataset (or downloads/parses it if implemented in your `MetaFinder2.py`), embeds with `nomic-embed-text`, and persists to **`journal_chroma_db/`**.

```bash
# Ensure Ollama is running locally first.
python MetaFinder2.py --build
```

- Expected outputs:
  - **`journal_data/`** — preprocessed text chunks/metadata.
  - **`journal_chroma_db/`** — ChromaDB persistent directory.

> Re-run `--build` whenever you update/replace the source data.

#### 6) Run the Streamlit app
```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`) and you’re ready to go.

---

## Typical Workflow

1. Paste your **manuscript title/abstract/keywords** into the Streamlit form.
2. Run **Vector Search** to see top journals from your local PubMed-derived index.
3. Run **LLM-Assisted Web Search** to brainstorm & validate additional venues via Tavily links.
4. Compare, filter, and shortlist venues.

---

## Configuration Notes

- **Ollama models**
  - Embeddings: `nomic-embed-text`
  - Reasoning/brainstorming: `gemma:2b`
- **Environment**
  - `TAVILY_API_KEY` must be present at runtime for the web search feature.
- **Persistence**
  - ChromaDB uses the `journal_chroma_db/` folder for persistence so you don’t recompute embeddings every run.

---

## Troubleshooting

- **Ollama connection errors**
  - Ensure the daemon is running: `ollama serve` (if not auto-started).
  - Verify models are available: `ollama list`.
- **Embedding/LLM model not found**
  - Run the pulls again:
    ```bash
    ollama pull nomic-embed-text
    ollama pull gemma:2b
    ```
- **No web results from LLM-assisted search**
  - Confirm `echo $TAVILY_API_KEY` prints a value (or `setx` on Windows).
- **Slow first query**
  - The first run warms caches and may trigger lazy loads; later queries should be faster.

---

## Roadmap (nice-to-have)

- Add per-discipline ranking heuristics (acceptance rate, SJR, open access flags).
- Export recommendations to Markdown/CSV.
- Add evaluation scripts & unit tests for ranking quality.

---

## License

Choose a license (e.g., MIT) and add a `LICENSE` file if you plan to open-source the project.

---

## Acknowledgments

- **Ollama** for local model serving.
- **LangChain** ecosystem for tool/chain integration.
- **Tavily** for reliable web search results.
