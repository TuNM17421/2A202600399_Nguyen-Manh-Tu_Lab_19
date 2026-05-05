"""
Graph RAG - QA Chain
Query Neo4j graph via LLM-generated Cypher → answer with GPT-4o-mini

Improvements v2:
  - Entity catalogue baked into Cypher prompt at build time (no runtime injection issue)
  - Graph triples converted to natural-language sentences before RAGAS
  - Entity expansion: 1-hop from initial results to enrich context
  - Proper 2-hop keyword fallback when Cypher fails
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from neo4j.exceptions import CypherSyntaxError

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j Cypher query writer.
Write a valid Cypher query that retrieves information to answer the question.

STRICT RULES:
1. Use ONLY node labels and relationship types listed in the schema below.
2. Relationships MUST use pattern: (a)-[:REL_TYPE]->(b)  -- never put REL_TYPE after a colon on a node.
3. To filter by name use: WHERE toLower(n.id) CONTAINS toLower('keyword')
4. Use OPTIONAL MATCH for secondary hops to avoid empty results.
5. Exclude Document nodes: AND NOT n:Document AND NOT m:Document
6. LIMIT to 25 rows. Read-only queries only (no MERGE, CREATE, DELETE).

RELATIONSHIP DIRECTIONS:
  (Company)-[:FOUNDED_BY]->(Person)
  (Company)-[:OWNED_BY]->(Company)
  (Company)-[:FUNDED_BY]->(Company)
  (Company)-[:PART_OF]->(Company)
  (Company)-[:DEVELOPED_BY]->(Model)
  (Person)-[:CEO_OF]->(Company)
  (Person)-[:WORKS_AT]->(Organization)
  (Model)-[:COMPARABLE_TO]->(Model)
  (Organization)-[:INVESTED_IN]->(Company)

KNOWN ENTITY NAMES (use these exact strings in CONTAINS filters):
{entities}

SCHEMA:
{schema}

EXAMPLES:
  Q: Who founded DeepSeek?
  MATCH (c:Company)-[:FOUNDED_BY]->(p:Person) WHERE toLower(c.id) CONTAINS 'deepseek' RETURN c.id AS company, p.id AS founder

  Q: What is the relationship between Anthropic and OpenAI?
  MATCH (a:Company)-[r]->(b) WHERE toLower(a.id) CONTAINS 'anthropic' AND NOT a:Document AND NOT b:Document RETURN a.id, type(r), b.id LIMIT 20

  Q: Who invested in OpenAI?
  MATCH (o:Organization)-[:INVESTED_IN]->(c:Company) WHERE toLower(c.id) CONTAINS 'openai' RETURN o.id AS investor, c.id AS company

QUESTION: {question}

Cypher Query (raw Cypher only -- no markdown, no explanation):"""

QA_TEMPLATE = """You are a helpful assistant. Use ONLY the information below to answer the question.
If the information is insufficient, say "I don't have enough information to answer this."

Information:
{context}

Question: {question}

Answer:"""


def get_graph() -> Neo4jGraph:
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    graph.refresh_schema()
    return graph


def _fetch_entity_catalogue(graph: Neo4jGraph) -> str:
    labels = ["Company", "Person", "Model", "Organization", "Technology"]
    lines = []
    for label in labels:
        rows = graph.query(f"MATCH (n:{label}) RETURN n.id AS id ORDER BY n.id")
        ids = [r["id"] for r in rows if r["id"]]
        if ids:
            lines.append(f"  {label}: {', '.join(ids)}")
    return "\n".join(lines)


def _triples_to_text(graph_result: list) -> list:
    sentences = []
    for row in graph_result:
        keys = set(row.keys())
        if {"from", "rel", "to"}.issubset(keys):
            rel_text = row["rel"].replace("_", " ").lower()
            sentences.append(f"{row['from']} {rel_text} {row['to']}.")
        else:
            parts = [f"{k}: {v}" for k, v in row.items() if v is not None]
            sentences.append(", ".join(parts) + ".")
    seen = set()
    unique = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _expand_entities(graph: Neo4jGraph, graph_result: list) -> list:
    entity_ids = set()
    for row in graph_result:
        for v in row.values():
            if isinstance(v, str) and v and len(v) < 100:
                entity_ids.add(v)
    expanded = []
    for eid in list(entity_ids)[:10]:
        try:
            rows = graph.query(
                "MATCH (n)-[r]->(m) WHERE n.id = $eid AND NOT n:Document AND NOT m:Document "
                "RETURN n.id AS `from`, type(r) AS rel, m.id AS `to` LIMIT 10",
                params={"eid": eid},
            )
            expanded.extend(rows)
        except Exception:
            pass
    return expanded


def _two_hop_fallback(graph: Neo4jGraph, question: str) -> list:
    keywords = [w for w in question.split() if len(w) > 4][:4]
    results = []
    for kw in keywords:
        try:
            rows = graph.query(
                "MATCH (n)-[r]->(m) WHERE toLower(n.id) CONTAINS toLower($kw) "
                "AND NOT n:Document AND NOT m:Document "
                "RETURN n.id AS `from`, type(r) AS rel, m.id AS `to` LIMIT 15",
                params={"kw": kw},
            )
            results.extend(rows)
            rows2 = graph.query(
                "MATCH (n)<-[r]-(m) WHERE toLower(n.id) CONTAINS toLower($kw) "
                "AND NOT n:Document AND NOT m:Document "
                "RETURN m.id AS `from`, type(r) AS rel, n.id AS `to` LIMIT 15",
                params={"kw": kw},
            )
            results.extend(rows2)
        except Exception:
            pass
    return results


def build_chain():
    graph = get_graph()
    entities = _fetch_entity_catalogue(graph)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Bake entities into template at build time so chain does not receive unknown input key
    filled_template = CYPHER_GENERATION_TEMPLATE.replace("{entities}", entities)

    cypher_prompt = PromptTemplate(
        template=filled_template,
        input_variables=["schema", "question"],
    )
    qa_prompt = PromptTemplate(
        template=QA_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        return_intermediate_steps=True,
        verbose=False,
        allow_dangerous_requests=True,
    )
    return chain, graph, entities


def ask(question: str, chain=None, graph=None, entities: str = "") -> dict:
    if chain is None or graph is None:
        chain, graph, entities = build_chain()

    cypher_query = ""
    contexts = []
    answer = "I don't have enough information to answer this."

    try:
        result = chain.invoke({"query": question})
        steps = result.get("intermediate_steps", [])
        cypher_query = steps[0].get("query", "") if steps else ""
        graph_result = steps[1].get("context", []) if len(steps) > 1 else []

        if graph_result:
            expanded = _expand_entities(graph, graph_result)
            all_rows = graph_result + expanded
        else:
            print(f"    [INFO] Cypher returned empty, running 2-hop fallback...")
            all_rows = _two_hop_fallback(graph, question)
            cypher_query += " [empty->fallback]"

        contexts = _triples_to_text(all_rows) if all_rows else []

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        ctx_text = "\n".join(contexts) if contexts else "no relevant graph data found"
        qa_msg = QA_TEMPLATE.format(context=ctx_text, question=question)
        answer = llm.invoke(qa_msg).content

    except (CypherSyntaxError, Exception) as e:
        print(f"    [WARN] Cypher error -> 2-hop fallback. {type(e).__name__}: {str(e)[:120]}")
        cypher_query = f"FALLBACK ({type(e).__name__})"
        fallback_rows = _two_hop_fallback(graph, question)
        contexts = _triples_to_text(fallback_rows) if fallback_rows else ["no graph data returned"]
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        ctx_text = "\n".join(contexts)
        qa_msg = QA_TEMPLATE.format(context=ctx_text, question=question)
        answer = llm.invoke(qa_msg).content

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts if contexts else ["no graph data returned"],
        "cypher": cypher_query,
    }


if __name__ == "__main__":
    print("=== Graph RAG Chain Test ===\n")
    chain, graph, entities = build_chain()

    questions = [
        "Who founded DeepSeek and what is their role at High-Flyer?",
        "Which company previously employed the founders of Anthropic?",
        "What is the relationship between xAI and SpaceX?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        out = ask(q, chain, graph, entities)
        print(f"A: {out['answer']}")
        print(f"Cypher: {out['cypher'][:120]}")
        print(f"Contexts ({len(out['contexts'])}): {out['contexts'][:3]}")
