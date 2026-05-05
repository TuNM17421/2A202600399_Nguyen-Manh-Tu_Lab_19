"""
Graph RAG - Ingest Pipeline
Load .md files → Extract entities & relationships via LLM → Store in Neo4j
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

# Entity/relation types to extract — scoped to our AI company domain
ALLOWED_NODES = [
    "Company", "Person", "Model", "Technology",
    "Organization", "Award", "License", "Product",
]
ALLOWED_RELATIONSHIPS = [
    "FOUNDED_BY", "OWNED_BY", "FUNDED_BY", "CEO_OF",
    "DEVELOPED_BY", "TRAINED_BY", "COMPARABLE_TO",
    "RELEASED_BY", "USES_TECHNOLOGY", "PART_OF",
    "COMPETED_WITH", "INVESTED_IN", "WORKS_AT",
]


def get_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )


def load_documents() -> list:
    docs = []
    for md_file in sorted(DATA_DIR.glob("*.md")):
        loader = TextLoader(str(md_file), encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = md_file.stem
        docs.extend(loaded)
        print(f"  Loaded: {md_file.name}")
    return docs


def split_documents(docs) -> list:
    # Larger chunks for graph extraction — more context = better entity linking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  Total chunks: {len(chunks)}")
    return chunks


def extract_and_store(chunks: list, graph: Neo4jGraph):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_NODES,
        allowed_relationships=ALLOWED_RELATIONSHIPS,
        node_properties=["description"],
        relationship_properties=["since", "detail"],
    )

    print(f"  Extracting graph from {len(chunks)} chunks (this may take a while)...")
    graph_docs = []
    for i, chunk in enumerate(chunks, 1):
        result = transformer.convert_to_graph_documents([chunk])
        graph_docs.extend(result)
        nodes = len(result[0].nodes) if result else 0
        rels = len(result[0].relationships) if result else 0
        print(f"    [{i}/{len(chunks)}] source={chunk.metadata.get('source','')} → {nodes} nodes, {rels} rels")

    total_nodes = sum(len(gd.nodes) for gd in graph_docs)
    total_rels = sum(len(gd.relationships) for gd in graph_docs)
    print(f"  Extracted: {total_nodes} nodes, {total_rels} relationships")

    graph.add_graph_documents(
        graph_docs,
        baseEntityLabel=True,
        include_source=True,
    )
    print("  Graph stored in Neo4j.")
    return graph_docs


def clear_graph(graph: Neo4jGraph):
    """Wipe all data in Neo4j — use before re-ingesting."""
    graph.query("MATCH (n) DETACH DELETE n")
    print("  Neo4j graph cleared.")


if __name__ == "__main__":
    print("=== Graph RAG Ingest ===")

    graph = get_graph()

    print("\n[0] Clearing existing graph data...")
    clear_graph(graph)

    print("\n[1] Loading documents...")
    docs = load_documents()

    print("\n[2] Splitting into chunks...")
    chunks = split_documents(docs)

    print("\n[3] Extracting entities & relationships → Neo4j...")
    extract_and_store(chunks, graph)

    print("\n[4] Graph schema:")
    graph.refresh_schema()
    print(graph.schema)

    print("\nDone. Open http://localhost:7474 to explore the graph.")
