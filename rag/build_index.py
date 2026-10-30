""" Turning the output from predictions into a searchable chroma vector database
Repo address: python rag/build_index.py
"""

import shutil
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PREDICTIONS_PATH = "results/predictions.csv"
DB_DIR = "rag/chroma_db"
COLLECTION = "absa_reviews"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 1000

def load_documents(path: str) -> list[Document]:
    """One Document per sentence: the text, plus its labels as metadata."""
    df = pd.read_csv(path)
    docs = []
    for row in df.itertuples(index=False):
        docs.append(Document(
            page_content=row.sentence,
            metadata={
                "sentence_id": int(row.sentence_id),
                "review_id": int(row.review_id),
                "rating": float(row.rating),
                "aspect": row.pred_aspect,          # model's guess
                "sentiment": row.pred_sentiment,    # model's guess
                "aspect_conf": float(row.aspect_conf),
                "sentiment_conf": float(row.sentiment_conf),
            },
        ))
    return docs

def main():
    docs = load_documents(PREDICTIONS_PATH)
    print(f"Loaded {len(docs)} sentences")

    # Start fresh every run so old data never mixes with new
    if Path(DB_DIR).exists():
        shutil.rmtree(DB_DIR)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )

    # Add in batches so we can see progress
    for start in range(0, len(docs), BATCH_SIZE):
        batch = docs[start:start + BATCH_SIZE]
        ids = [str(d.metadata["sentence_id"]) for d in batch]
        db.add_documents(batch, ids=ids)
        print(f"  Indexed {start + len(batch)}/{len(docs)}")

    print(f"Done. Database saved to {DB_DIR}")


if __name__ == "__main__":
    main()