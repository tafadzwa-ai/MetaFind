import os 

#data locations 

data_dir = os.environ.get('JF_Data_Dir','./data') #holds both research papers and the journals for error handling the environmental variable JF_Data_Dir if it exists otherwise defaults to ./data
papers_CSV = os.path.join(data_dir,'papers.csv') # take these folder/file names and glue them together into one valid path for my operating system
journals_CSV = os.path.join(data_dir,'journals.csv') 

# PUBMED API

email = os.environ.get('NCBI_email','w8a35@keele.ac.uk') #email to use with NCBI API
api_key = os.environ.get('NCBI_api_key',None) #api key to use with NCBI API

Topics = [
    '(bioinformatics[Title/Abstract])',
    '("human-computer interaction"[Title/Abstract] or HCI[Title/Abstract])',
    '(cybersecurity[Title/Abstract]) or security[Title/Abstract]',
    '(artificial intelligence[Title/Abstract] or "machine learning"[Title/Abstract] or "deep learning"[Title/Abstract]) or AI[Title/Abstract]',
    '(data science[Title/Abstract] or "data analytics"[Title/Abstract] or "big data"[Title/Abstract])',
    '(software engineering[Title/Abstract])'
] # search queries for PubMed API tells the program what topics to search for and only the abstract and title. 


date_range = '2021:3000' # range of publication dates to search for in PubMed API, format is YYYY:YYYY

MAX_PMIDS = 1500 # maximum number of PMIDs to retrieve from PubMed API per topic
BATCH_SIZE = 200 # number of PMIDs to fetch in each batch from PubMed API, max is 200 according to pubmed rules

Chroma_Dir = os.environ.get('JF_Chroma_Dir','./chroma_store') # directory to store ChromaDB database, can be set with JF_Chroma_Dir environmental variable otherwise defaults to ./chroma_store
Embed_Model = os.environ.get('JF_Embed_Model','bge-small-en') # embedding model to use, can be set with JF_Embed_Model environmental variable otherwise defaults to 'bge-small-en'
Gen_Model = os.environ.get('JF_GEN_MODEL', 'llama3.1:8b-instruct') #via Ollama 3.1 8b instruct model, can be set with JF_GEN_MODEL environmental variable otherwise defaults to 'llama3.1:8b-instruct'

K_retrieval = int(os.environ.get('JF_K','60')) #when i search for similar papers retrieve the top 60 papers 
Top_Venues = int(os.environ.get('JF_Top','5')) # number of top venues to include in the report, can be set with JF_Top environmental variable otherwise defaults to 5
Evidence_K = int(os.environ.get('JF_evid','3')) #for every recommended venue show me the top 3 evidence papers

