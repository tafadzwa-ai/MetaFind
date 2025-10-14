import os,csv,time
from typing import List,Dict
from dotenv import load_dotenv

load_dotenv()

ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL","you@example.com")
ENTREZ_TOOL = os.getenv("ENTREZ_TOOL","MetaFind")

DATA_DIR = "./dataset"
WORKS_CSV = os.path.join(DATA_DIR,"works.csv")
VENUES_CSV = os.path.join(DATA_DIR,"venues.csv")

CHROMA_DIR   = os.getenv("JF_CHROMA_DIR", "./chroma_store")
EMBED_MODEL  = os.getenv("JF_EMBED_MODEL", "bge-small-en")
GEN_MODEL    = os.getenv("JF_GEN_MODEL", "llama3.1:8b-instruct")

TOP_K        = int(os.getenv("JF_TOP_K", "60"))
TOP_VENUES   = int(os.getenv("JF_TOP_VENUES", "5"))
EVIDENCE_K   = int(os.getenv("JF_EVIDENCE_K", "3"))


from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from Bio import Entrez
Entrez.email = ENTREZ_EMAIL
Entrez.tool = ENTREZ_TOOL

USE_OPENAI=bool(os.getenv('OPENAI_API_KEY'))
if USE_OPENAI:
    from langchain_openai import ChatOpenAI
else:
        from langchain_community.chat_models import ChatOllama


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

def info(msg): print(f"[Info] {msg}")
def ok(msg):   print(f"[OK] {msg}")


#Fetch DataSet

TOPICS = [
    '(bioinformatics[Title/Abstract])',
    '("human-computer interaction"[Title/Abstract] OR HCI[Title/Abstract])',
    '(cybersecurity[Title/Abstract])',
    '("artificial intelligence"[Title/Abstract] OR "machine learning"[Title/Abstract] OR "deep learning"[Title/Abstract])'
]
MAX_PER_TOPIC = 120    # keep small to stay beginner-friendly; increase later
DATE_FILTER   = '2021:3000'

def search_pmids(query: str, retmax: int) -> List[str]:
    h = Entrez.esearch(db="pubmed", term=f"({query}) AND ({DATE_FILTER}[DP])", retmax=retmax)
    rec = Entrez.read(h); h.close()
    return rec.get("IdList", [])

def fetch_details(pmids: List[str]) -> List[Dict]:
    if not pmids: return []
    h = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="medline", retmode="xml")
    data = Entrez.read(h); h.close()
    out = []
    for a in data.get("PubmedArticle", []):
        art = a["MedlineCitation"]["Article"]
        pmid = a["MedlineCitation"]["PMID"]
        title = str(art.get("ArticleTitle","")) or ""
        abstract = ""
        if art.get("Abstract"):
            abstract = " ".join([ab.get("AbstractText","") for ab in art["Abstract"].get("AbstractText", [])])
        year = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {}).get("Year", "")
        journal = art.get("Journal", {}).get("Title", "")

        kws = []
        for kw in (art.get("KeywordList") or []):
            for k in kw: kws.append(str(k))
        mesh = a["MedlineCitation"].get("MeshHeadingList", [])
        for m in mesh:
            if m.get("DescriptorName"): kws.append(str(m["DescriptorName"]))

        authors = []
        for au in art.get("AuthorList", []):
            last = au.get("LastName",""); fore = au.get("ForeName","")
            if last or fore: authors.append(f"{last} {fore}".strip())

        out.append({
            "pubmed_id": str(pmid),
            "title": title,
            "abstract": abstract,
            "keywords": "|".join(sorted(set(kws))),
            "authors": "|".join(authors),
            "year": str(year),
            "journal": str(journal)
        })
    return out

def cmd_fetch():
    ensure_dirs()
    all_rows = []
    for q in TOPICS:
        info(f"Querying: {q}")
        pmids = search_pmids(q, MAX_PER_TOPIC)
        info(f"  PMIDs found: {len(pmids)}")
        time.sleep(0.4)
        rows = fetch_details(pmids)
        for r in rows:
            r["category"] = q   # soft label from query
        all_rows.extend(rows)
        time.sleep(0.4)

    with open(WORKS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pubmed_id","title","abstract","keywords","authors","year","journal","category"])
        w.writeheader(); w.writerows(all_rows)

    venues = {}
    for r in all_rows:
        j = (r["journal"] or "Unknown").strip()
        if j not in venues:
            venues[j] = {"venue_id": j, "display_name": j, "type": "journal"}

    with open(VENUES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["venue_id","display_name","type"])
        w.writeheader(); w.writerows(venues.values())

    ok(f"Saved {len(all_rows)} works → {WORKS_CSV}")
    ok(f"Saved {len(venues)} venues → {VENUES_CSV}")
