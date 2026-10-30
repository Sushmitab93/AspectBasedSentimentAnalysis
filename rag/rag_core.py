"""Core RAG logic: filter by model labels, search by meaning, answer with citations."""
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()   # reads GROQ_API_KEY and GROQ_MODEL from .env

DB_DIR = "rag/chroma_db"
COLLECTION = "absa_reviews"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ASPECTS = ["battery", "camera", "connectivity", "design",
           "display", "performance", "price", "sound"]
SENTIMENTS = ["negative", "positive"]

_db = None   # loaded once, then reused


def get_db() -> Chroma:
    global _db
    if _db is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        _db = Chroma(collection_name=COLLECTION, embedding_function=embeddings,
                     persist_directory=DB_DIR)
    return _db


# ---------- Step A: build the filter from the user's choices ----------
def build_filter(aspect=None, sentiment=None, min_conf=0.0):
    rules = []
    if aspect:
        rules.append({"aspect": aspect})
    if sentiment:
        rules.append({"sentiment": sentiment})
    if min_conf > 0:
        rules.append({"sentiment_conf": {"$gte": min_conf}})

    if not rules:
        return None            # no filter: search everything
    if len(rules) == 1:
        return rules[0]        # Chroma wants a single rule on its own
    return {"$and": rules}     # 2+ rules must be wrapped in $and


# ---------- Step B: retrieve matching sentences ----------
def retrieve(question, aspect=None, sentiment=None, min_conf=0.0, k=12):
    return get_db().similarity_search(
        question, k=k, filter=build_filter(aspect, sentiment, min_conf)
    )


# ---------- Step C: number the sentences for the LLM ----------
def format_context(docs):
    lines = []
    for i, d in enumerate(docs, start=1):
        stars = int(d.metadata["rating"])
        lines.append(f"[{i}] ({stars}-star review) {d.page_content}")
    return "\n".join(lines)


# ---------- Step D: the instructions for the LLM ----------
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You summarize customer review sentences for a product team.\n"
     "Rules:\n"
     "1. Use ONLY the numbered sentences provided. Never add outside knowledge.\n"
     "2. Some sentences may be mislabeled or off-topic. Ignore any sentence that "
     "does not match the question (for example, praise when the question asks "
     "about complaints).\n"
     "3. Answer as 3 to 6 short bullet points, grouping similar points together.\n"
     "4. End every bullet with the numbers of the sentences that support it, "
     "like [1][4].\n"
     "5. If no sentence answers the question, say so plainly."),
    ("human", "Question: {question}\n\nReview sentences:\n{context}"),
])


# ---------- Step E: pick the LLM ----------
def get_llm(provider="groq"):
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL"), temperature=0)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1"), temperature=0)
    raise ValueError(f"Unknown provider: {provider}")


# ---------- Put it all together ----------
def answer(question, aspect=None, sentiment=None, min_conf=0.0, k=12, provider="groq"):
    docs = retrieve(question, aspect, sentiment, min_conf, k)
    if not docs:
        return {"answer": "No sentences match these filters.", "sources": []}

    chain = PROMPT | get_llm(provider)
    response = chain.invoke({"question": question, "context": format_context(docs)})

    sources = [
        {"n": i, "sentence_id": d.metadata["sentence_id"],
         "rating": d.metadata["rating"], "aspect": d.metadata["aspect"],
         "sentiment": d.metadata["sentiment"], "text": d.page_content}
        for i, d in enumerate(docs, start=1)
    ]
    return {"answer": response.content, "sources": sources}