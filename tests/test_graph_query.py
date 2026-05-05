import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx

from src.graph_query import (
    load_graph,
    extract_entity_from_question,
    get_k_hop_subgraph_nodes,
    collect_edges_from_nodes,
    textualize_edges,
)


def test_load_graph():
    graph = load_graph()

    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0


def test_extract_entity_exact_match():
    graph = load_graph()

    entity = extract_entity_from_question(
        "Who founded OpenAI?",
        graph
    )

    assert entity is not None
    assert entity.lower() == "openai"


def test_extract_entity_google():
    graph = load_graph()

    entity = extract_entity_from_question(
        "How is Google related to AlphaGo?",
        graph
    )

    assert entity is not None
    assert entity.lower() in ["google", "alphago"]


def test_get_2_hop_subgraph_nodes():
    graph = load_graph()

    nodes = get_k_hop_subgraph_nodes(
        graph=graph,
        start_node="OpenAI",
        hops=2
    )

    assert "OpenAI" in nodes
    assert len(nodes) >= 1


def test_collect_edges_from_nodes():
    graph = load_graph()

    nodes = get_k_hop_subgraph_nodes(
        graph=graph,
        start_node="OpenAI",
        hops=2
    )

    edges = collect_edges_from_nodes(graph, nodes)

    assert isinstance(edges, list)
    assert len(edges) > 0

    for source, relation, target in edges:
        assert isinstance(source, str)
        assert isinstance(relation, str)
        assert isinstance(target, str)


def test_textualize_edges():
    edges = [
        ("OpenAI", "FOUNDED_BY", "Sam Altman"),
        ("Microsoft", "INVESTED_IN", "OpenAI"),
    ]

    context = textualize_edges(edges)

    assert "OpenAI --FOUNDED_BY--> Sam Altman" in context
    assert "Microsoft --INVESTED_IN--> OpenAI" in context


def test_textualize_empty_edges():
    context = textualize_edges([])

    assert context == "No graph context found."