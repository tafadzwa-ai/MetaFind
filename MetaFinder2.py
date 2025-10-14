import os
import csv
import time
import collections
import re
from typing import List

# --- All your imports go here ---
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Bio import Entrez

# --- Configuration ---
ENTREZ_EMAIL = "t.w.machemedze@gmail.com"
ENTREZ_TOOL = "VenueRecommender"
TAVILY_API_KEY = "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"
DATA_DIR = "./journal_data"
WORKS_CSV = os.path.join(DATA_DIR, "works.csv")
CHROMA_DIR = "./journal_chroma_db"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "gemma:2b"
TOP_K_RESULTS = 50
TOP_N_VENUES = 8

class VenueRecommender:
    """
    Backend logic for the MetaFind app.
    """
    def __init__(self):
        Entrez.email = ENTREZ_EMAIL
        Entrez.tool = ENTREZ_TOOL
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(CHROMA_DIR, exist_ok=True)
        print("🚀 MetaFinder Recommender Initialized.")

    def recommend_from_index(self, query_abstract: str) -> str:
        """Recommends venues with evidence and PubMed links."""
        output_string = "### From Vector Index\n\n"
        if not os.path.isdir(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
            return "[Error] Chroma database not found."

        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        results = vector_store.similarity_search(query_abstract, k=TOP_K_RESULTS)

        if not results:
            return output_string + "Could not find any similar articles in the index."

        journal_counts = collections.Counter(doc.metadata.get("journal", "Unknown") for doc in results)
        
        for i, (journal, count) in enumerate(journal_counts.most_common(TOP_N_VENUES), 1):
            output_string += f"{i}. **{journal}** (Found in {count} similar articles)\n"
            evidence_count = 0
            for doc in results:
                if doc.metadata.get("journal") == journal and evidence_count < 2:
                    title = doc.metadata.get('title', 'No Title')
                    pmid = doc.metadata.get('pubmed_id', '')
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    output_string += f"   - *Evidence:* [{title[:70]}...]({url}) (PMID: {pmid})\n"
                    evidence_count += 1
        
        return output_string

    def recommend_with_llm(self, query_abstract: str) -> str:
        """
        Uses an LLM to generate article ideas, then uses Python to find the links.
        """
        output_string = "### Recommendations from LLM-Assisted Search\n\n"
        
        if not TAVILY_API_KEY or TAVILY_API_KEY == "tvly-your-key-here":
            return "[Error] TAVILY_API_KEY is not set correctly."
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

        try:
            # === Step 1: Ask the LLM for a simple list of article titles ===
            print("--- Asking LLM for article ideas... ---")
            prompt = ChatPromptTemplate.from_template(
                "Based on the abstract below, generate a list of 5 relevant academic article titles. "
                "Your answer MUST be ONLY the titles, each on a new line. Do not use numbers, commas, or bullet points."
                "\n\nABSTRACT: {abstract}"
            )
            llm = ChatOllama(model=GEN_MODEL)
            output_parser = StrOutputParser()
            chain = prompt | llm | output_parser
            
            raw_titles = chain.invoke({"abstract": query_abstract})
            
            # === Step 2: More robust parsing of the LLM's output ===
            # Split by newline and clean up each line
            article_titles = []
            for line in raw_titles.split('\n'):
                # Remove leading/trailing whitespace and common markdown characters
                cleaned_line = line.strip().lstrip('*- ')
                if cleaned_line: # Only add non-empty lines
                    article_titles.append(cleaned_line)

            if not article_titles:
                return output_string + "The LLM could not generate any valid article titles."

            # === Step 3: Use Python to find the link for each title ===
            print("--- Finding links for each title using Tavily... ---")
            search_tool = TavilySearchResults(max_results=1)
            
            output_string += "**Recommended Articles:**\n"
            for i, title in enumerate(article_titles, 1):
                print(f"   Searching for: '{title}'")
                # Use the search tool directly from Python
                search_results = search_tool.invoke(title)
                
                if search_results and 'url' in search_results[0]:
                    url = search_results[0]['url']
                    output_string += f"{i}. **[{title}]({url})**\n"
                else:
                    output_string += f"{i}. **{title}** (Could not find a direct link)\n"

            return output_string

        except Exception as e:
            return output_string + f"[Error] Could not get a response from the LLM. Is Ollama running?\nDetails: {e}"

