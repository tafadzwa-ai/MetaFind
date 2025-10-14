SYSTEM = "You are a research assistant. Suggest 3–5 venues for the abstract (with short reasons)."
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user", "Title: {title}\n\nAbstract: {abstract}\n\nKeywords: {keywords}")
])

def cmd_prompt(abstract: str):
    if USE_OPENAI:
        chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    else:
        chat = ChatOllama(model=GEN_MODEL, temperature=0.2)

    msg = PROMPT.format_messages(title="Untitled", abstract=abstract, keywords="")
    out = chat.invoke(msg).content
    print("\n=== Prompt-only LLM Recommender ===\n")
    print(out)
    print()
