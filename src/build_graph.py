import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


TRIPLES_PATH = Path("outputs/triples.json")
GRAPH_IMAGE_PATH = Path("outputs/graph.png")
GRAPH_DATA_PATH = Path("outputs/graph.gml")


def load_triples():
    if not TRIPLES_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {TRIPLES_PATH}. Please run src/extract_triples.py first."
        )

    with open(TRIPLES_PATH, "r", encoding="utf-8") as f:
        triples = json.load(f)

    return triples


def build_graph(triples):
    graph = nx.DiGraph()

    for triple in triples:
        subject = triple.get("subject", "").strip()
        relation = triple.get("relation", "").strip()
        obj = triple.get("object", "").strip()

        if not subject or not relation or not obj:
            continue

        graph.add_node(subject)
        graph.add_node(obj)
        graph.add_edge(subject, obj, relation=relation)

    return graph


def save_graph_data(graph):
    GRAPH_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(graph, GRAPH_DATA_PATH)


def draw_graph(graph):
    GRAPH_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(22, 16))

    pos = nx.spring_layout(
        graph,
        k=1.2,
        iterations=100,
        seed=42
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=1800,
        alpha=0.9
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=1.4,
        alpha=0.7
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=9,
        font_weight="bold"
    )

    edge_labels = nx.get_edge_attributes(graph, "relation")

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        rotate=False
    )

    plt.title("Tech Company Knowledge Graph", fontsize=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(GRAPH_IMAGE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def print_graph_summary(graph):
    print("Knowledge Graph Summary")
    print("=======================")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    print("\nSample nodes:")
    for node in list(graph.nodes())[:10]:
        print(f"- {node}")

    print("\nSample edges:")
    for source, target, data in list(graph.edges(data=True))[:10]:
        relation = data.get("relation", "RELATED_TO")
        print(f"- {source} --{relation}--> {target}")


def main():
    triples = load_triples()
    graph = build_graph(triples)

    save_graph_data(graph)
    draw_graph(graph)
    print_graph_summary(graph)

    print(f"\nGraph image saved to: {GRAPH_IMAGE_PATH}")
    print(f"Graph data saved to: {GRAPH_DATA_PATH}")


if __name__ == "__main__":
    main()