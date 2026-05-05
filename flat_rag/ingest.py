"""
Flat RAG - Ingest Pipeline
Load .md files → Chunk → Embed → Persist to ChromaDB
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Disable ChromaDB telemetry to suppress posthog warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db" / "flat_rag"
COLLECTION_NAME = "flat_rag_docs"


def load_documents():
    docs = []
    for md_file in sorted(DATA_DIR.glob("*.md")):
        loader = TextLoader(str(md_file), encoding="utf-8")
        loaded = loader.load()
        # Tag source metadata
        for doc in loaded:
            doc.metadata["source"] = md_file.stem
        docs.extend(loaded)
        print(f"  Loaded: {md_file.name} ({len(loaded[0].page_content)} chars)")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"\n  Total chunks: {len(chunks)}")
    return chunks


def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"  Vectorstore persisted at: {CHROMA_DIR}")
    return vectorstore


def load_vectorstore():
    """Load existing vectorstore (no re-embed)."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


if __name__ == "__main__":
    print("=== Flat RAG Ingest ===")
    print("\n[1] Loading documents...")
    docs = load_documents()

    print("\n[2] Splitting into chunks...")
    chunks = split_documents(docs)

    print("\n[3] Embedding & persisting to ChromaDB...")
    build_vectorstore(chunks)

    print("\nDone.")
