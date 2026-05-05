import os
import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

CORPUS_PATH = Path("data/tech_company_corpus.txt")
INDEX_PATH = Path("outputs/flat_rag.index")
CHUNKS_PATH = Path("outputs/flat_rag_chunks.json")


def load_corpus() -> str:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {CORPUS_PATH}. Please create the corpus file first."
        )

    return CORPUS_PATH.read_text(encoding="utf-8")


def split_corpus_into_chunks(text: str) -> List[str]:
    """
    Split corpus by paragraphs.
    For this lab corpus, each paragraph is already a meaningful chunk.
    """
    chunks = []

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph:
            chunks.append(paragraph)

    return chunks


def get_embedding(text: str) -> List[float]:
    """
    Create embedding vector for one text.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding


def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Create embeddings for multiple texts.
    Returns numpy array with shape: (num_texts, embedding_dim)
    """
    embeddings = []

    for text in texts:
        embedding = get_embedding(text)
        embeddings.append(embedding)

    array = np.array(embeddings).astype("float32")

    # Normalize vectors for cosine similarity using inner product
    faiss.normalize_L2(array)

    return array


def build_faiss_index(chunks: List[str]) -> faiss.IndexFlatIP:
    """
    Build FAISS index using cosine similarity.
    """
    embeddings = get_embeddings(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def save_index_and_chunks(index: faiss.IndexFlatIP, chunks: List[str]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))

    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def load_index_and_chunks():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Flat RAG index not found. Please run build_flat_rag_index() first."
        )

    index = faiss.read_index(str(INDEX_PATH))

    chunks = json.loads(
        CHUNKS_PATH.read_text(encoding="utf-8")
    )

    return index, chunks


def build_flat_rag_index() -> None:
    corpus = load_corpus()
    chunks = split_corpus_into_chunks(corpus)

    print(f"Loaded corpus.")
    print(f"Total chunks: {len(chunks)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    index = build_faiss_index(chunks)
    save_index_and_chunks(index, chunks)

    print("Flat RAG index built successfully.")
    print(f"Index saved to: {INDEX_PATH}")
    print(f"Chunks saved to: {CHUNKS_PATH}")


def retrieve_chunks(question: str, top_k: int = 3) -> List[Dict[str, Any]]:
    index, chunks = load_index_and_chunks()

    query_embedding = np.array([get_embedding(question)]).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append({
            "chunk_id": int(idx),
            "score": float(score),
            "text": chunks[idx]
        })

    return results


def build_context_from_chunks(retrieved_chunks: List[Dict[str, Any]]) -> str:
    if not retrieved_chunks:
        return "No relevant context found."

    context_parts = []

    for item in retrieved_chunks:
        chunk_id = item["chunk_id"]
        score = item["score"]
        text = item["text"]

        context_parts.append(
            f"[Chunk {chunk_id} | Score: {score:.4f}]\n{text}"
        )

    return "\n\n".join(context_parts)


def answer_with_flat_rag(question: str, context: str) -> str:
    prompt = f"""
You are a Flat RAG question answering assistant.

Use ONLY the retrieved context below to answer the user's question.
If the retrieved context does not contain enough information, answer:
"Not enough information in the retrieved context."

Retrieved context:
{context}

User question:
{question}

Answer clearly and concisely.
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You answer questions using only the retrieved text context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def query_flat_rag(question: str, top_k: int = 3, verbose: bool = True) -> Dict[str, Any]:
    retrieved_chunks = retrieve_chunks(question, top_k=top_k)
    context = build_context_from_chunks(retrieved_chunks)
    answer = answer_with_flat_rag(question, context)

    result = {
        "question": question,
        "top_k": top_k,
        "retrieved_chunks": retrieved_chunks,
        "context": context,
        "answer": answer
    }

    if verbose:
        print("\n================ Flat RAG Query ================")
        print(f"Question: {question}")
        print(f"Top K: {top_k}")

        print("\n---------------- Retrieved Context ----------------")
        print(context)

        print("\n---------------- Answer ----------------")
        print(answer)
        print("==================================================\n")

    return result


def interactive_loop():
    print("Flat RAG Query")
    print("==============")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("Bye.")
            break

        if not question:
            continue

        query_flat_rag(question, top_k=3, verbose=True)


if __name__ == "__main__":
    build_flat_rag_index()
    interactive_loop()