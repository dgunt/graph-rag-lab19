import os
from pathlib import Path
from difflib import get_close_matches
from typing import List, Tuple, Set

import networkx as nx
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GRAPH_PATH = Path("outputs/graph.gml")


def load_graph() -> nx.DiGraph:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {GRAPH_PATH}. Please run src/build_graph.py first."
        )

    graph = nx.read_gml(GRAPH_PATH)
    return graph


def extract_entity_from_question(question: str, graph: nx.DiGraph) -> str | None:
    """
    Extract the most likely entity from the question.

    Strategy:
    1. Exact substring match with graph nodes.
    2. Fuzzy match using difflib.
    3. Return None if no good match.
    """
    question_lower = question.lower()
    nodes = list(graph.nodes())

    # 1. Exact substring match
    exact_matches = []

    for node in nodes:
        node_text = str(node)
        if node_text.lower() in question_lower:
            exact_matches.append(node_text)

    if exact_matches:
        # Prefer the longest entity name
        return max(exact_matches, key=len)

    # 2. Fuzzy match
    possible_matches = get_close_matches(
        question,
        nodes,
        n=1,
        cutoff=0.45
    )

    if possible_matches:
        return possible_matches[0]

    return None


def get_k_hop_subgraph_nodes(
    graph: nx.DiGraph,
    start_node: str,
    hops: int = 2
) -> Set[str]:
    """
    Get all nodes within k hops from the start node.
    Includes both outgoing and incoming edges.
    """
    visited = {start_node}
    frontier = {start_node}

    for _ in range(hops):
        next_frontier = set()

        for node in frontier:
            if node not in graph:
                continue

            # Outgoing neighbors: node -> neighbor
            next_frontier.update(graph.successors(node))

            # Incoming neighbors: neighbor -> node
            next_frontier.update(graph.predecessors(node))

        next_frontier = next_frontier - visited
        visited.update(next_frontier)
        frontier = next_frontier

    return visited


def collect_edges_from_nodes(
    graph: nx.DiGraph,
    nodes: Set[str]
) -> List[Tuple[str, str, str]]:
    """
    Collect all edges where both source and target are inside selected nodes.
    Returns list of (source, relation, target).
    """
    edges = []

    for source, target, data in graph.edges(data=True):
        if source in nodes and target in nodes:
            relation = data.get("relation", "RELATED_TO")
            edges.append((source, relation, target))

    return edges


def textualize_edges(edges: List[Tuple[str, str, str]]) -> str:
    """
    Convert graph edges into readable text context.
    """
    if not edges:
        return "No graph context found."

    lines = []

    for source, relation, target in edges:
        line = f"{source} --{relation}--> {target}"
        lines.append(line)

    return "\n".join(lines)


def answer_with_graph_context(question: str, graph_context: str) -> str:
    prompt = f"""
You are a GraphRAG question answering assistant.

Use ONLY the graph context below to answer the user's question.
If the graph context does not contain enough information, answer:
"Not enough information in the graph context."

Graph context:
{graph_context}

User question:
{question}

Answer clearly and concisely.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You answer questions using only the provided knowledge graph context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def query_graph_rag(question: str, hops: int = 2, verbose: bool = True) -> dict:
    graph = load_graph()

    entity = extract_entity_from_question(question, graph)

    if entity is None:
        return {
            "question": question,
            "entity": None,
            "graph_context": "",
            "answer": "Could not identify a relevant entity in the graph."
        }

    subgraph_nodes = get_k_hop_subgraph_nodes(
        graph=graph,
        start_node=entity,
        hops=hops
    )

    edges = collect_edges_from_nodes(graph, subgraph_nodes)
    graph_context = textualize_edges(edges)

    answer = answer_with_graph_context(question, graph_context)

    result = {
        "question": question,
        "entity": entity,
        "hops": hops,
        "num_context_nodes": len(subgraph_nodes),
        "num_context_edges": len(edges),
        "graph_context": graph_context,
        "answer": answer
    }

    if verbose:
        print("\n================ GraphRAG Query ================")
        print(f"Question: {question}")
        print(f"Detected entity: {entity}")
        print(f"Hops: {hops}")
        print(f"Context nodes: {len(subgraph_nodes)}")
        print(f"Context edges: {len(edges)}")

        print("\n---------------- Graph Context ----------------")
        print(graph_context)

        print("\n---------------- Answer ----------------")
        print(answer)
        print("================================================\n")

    return result


def interactive_loop():
    print("GraphRAG 2-hop Query")
    print("====================")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("Bye.")
            break

        if not question:
            continue

        query_graph_rag(question, hops=2, verbose=True)


if __name__ == "__main__":
    interactive_loop()