# LAB DAY 19 - GraphRAG with Tech Company Corpus

## Student Information

- Name: Vũ Đức Minh
- Topic: GraphRAG vs Flat RAG Benchmark
- Environment:
  - Python 3.12
  - OpenAI API
  - NetworkX
  - FAISS
  - Matplotlib

---

# 1. Objective

The objective of this lab is to build and compare two retrieval systems:

1. Flat RAG using vector similarity search
2. GraphRAG using knowledge graph traversal

The project explores how graph-based reasoning differs from traditional vector-based retrieval systems.

---

# 2. Dataset

The dataset is a custom "Tech Company Corpus" containing information about:
- AI companies
- founders
- acquisitions
- products
- cloud platforms
- AI models
- technology ecosystems

Example entities:
- OpenAI
- Google
- DeepMind
- Microsoft
- NVIDIA
- Anthropic

Example relationships:
- FOUNDED_BY
- ACQUIRED_BY
- DEVELOPED
- INVESTED_IN
- INTEGRATED

---

# 3. System Architecture

## 3.1 Flat RAG Pipeline

Corpus
→ Chunking
→ OpenAI Embeddings
→ FAISS Vector Store
→ Similarity Retrieval
→ LLM Answer Generation

---

## 3.2 GraphRAG Pipeline

Corpus
→ Triple Extraction
→ Knowledge Graph Construction
→ Graph Traversal (2-hop)
→ Graph Context Generation
→ LLM Answer Generation

---

# 4. Triple Extraction

The system uses OpenAI models to extract knowledge graph triples from raw text.

Example:

Input:

OpenAI was founded by Sam Altman in 2015.

Output:

(OpenAI, FOUNDED_BY, Sam Altman)
(OpenAI, FOUNDED_IN, 2015)

---

# 5. Knowledge Graph Construction

The extracted triples are stored using NetworkX as a directed graph.

Nodes represent:
- companies
- founders
- products
- technologies

Edges represent:
- acquisitions
- investments
- product development
- organizational relationships

---

# 6. Graph Visualization

The graph was visualized using Matplotlib and NetworkX.

## Output Files

- outputs/graph.png
- outputs/subgraph_openai.png

---

# 7. Querying Process

## Flat RAG

1. Embed the question
2. Retrieve top-k chunks using FAISS
3. Generate answer using retrieved text context

---

## GraphRAG

1. Extract main entity from question
2. Traverse graph within 2-hop distance
3. Convert graph edges into textual context
4. Generate answer using graph context

---

# 8. Benchmark Setup

Total benchmark questions: 20

Question types:
- 1-hop factual questions
- 2-hop relational reasoning questions

Example:

How is Google related to AlphaGo?

Expected reasoning:

Google → acquired DeepMind
DeepMind → developed AlphaGo

---

# 9. Benchmark Results

| System | Accuracy | Avg Query Time |
|---|---|---|
| Flat RAG | 0.85 | 1.6717s |
| GraphRAG | 0.70 | 0.9067s |

---

# 10. Analysis

## Why Flat RAG Performed Better

The dataset contains compact paragraphs where most facts are already located within single chunks.

Therefore:
- Flat RAG can retrieve complete textual information directly
- GraphRAG loses some semantic richness during triple extraction

Additionally:
- Most benchmark questions are 1-hop factual questions
- The graph traversal logic is currently simple

---

## Strengths of GraphRAG

GraphRAG demonstrated advantages in:
- relational reasoning
- graph traversal
- lower query latency
- structured knowledge representation

GraphRAG performed better on some multi-hop reasoning questions such as:

How is Google related to AlphaGo?

---

# 11. Limitations

Current limitations include:
- simple entity extraction
- limited graph traversal strategy
- small benchmark dataset
- graph context textualization quality

---

# 12. Future Improvements

Potential future improvements:
- LLM-based entity extraction
- multi-entity reasoning
- shortest-path graph traversal
- hybrid retrieval (vector + graph)
- larger and more complex datasets

---

# 13. Conclusion

This lab demonstrates the differences between Flat RAG and GraphRAG systems.

Flat RAG achieved higher accuracy on the current benchmark because the dataset was relatively simple and highly information-dense.

GraphRAG showed strong potential for:
- multi-hop reasoning
- structured knowledge traversal
- scalable relationship-based retrieval

The experiment highlights that GraphRAG is not always superior to Flat RAG, but becomes increasingly valuable for complex relational retrieval tasks.

---

# 14. Deliverables

Included deliverables:
- Python source code
- Graph visualization images
- Benchmark CSV/JSON results
- Cost analysis report
- Benchmark evaluation pipeline