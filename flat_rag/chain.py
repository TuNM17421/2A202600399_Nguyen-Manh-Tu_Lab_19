"""
Flat RAG - QA Chain
Build retriever from ChromaDB + RetrievalQA chain with GPT-4o-mini
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from flat_rag.ingest import load_vectorstore

load_dotenv()

PROMPT_TEMPLATE = """You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""


def build_chain(top_k: int = 1):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain


def ask(question: str, chain=None) -> dict:
    """
    Returns:
        {
            "question": str,
            "answer": str,
            "contexts": list[str],   # raw chunk texts (for RAGAS)
            "sources": list[str],    # source file names
        }
    """
    if chain is None:
        chain = build_chain()

    result = chain.invoke({"query": question})

    contexts = [doc.page_content for doc in result["source_documents"]]
    sources = [doc.metadata.get("source", "") for doc in result["source_documents"]]

    return {
        "question": question,
        "answer": result["result"],
        "contexts": contexts,
        "sources": sources,
    }


if __name__ == "__main__":
    print("=== Flat RAG Chain Test ===\n")
    chain = build_chain()

    q = "Who founded DeepSeek and what is their role at High-Flyer?"
    print(f"Q: {q}")
    out = ask(q, chain)
    print(f"A: {out['answer']}")
    print(f"Sources: {list(set(out['sources']))}")
