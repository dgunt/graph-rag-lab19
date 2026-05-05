import json
import time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from src.flat_rag import query_flat_rag
from src.graph_query import query_graph_rag


BENCHMARK_PATH = Path("data/benchmark_questions.json")
OUTPUT_JSON_PATH = Path("outputs/benchmark_results.json")
OUTPUT_CSV_PATH = Path("outputs/benchmark_results.csv")
CACHE_PATH = Path("outputs/benchmark_cache.json")


def load_benchmark_questions() -> List[Dict[str, Any]]:
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {BENCHMARK_PATH}. Please create benchmark_questions.json first."
        )

    return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))


def load_cache() -> Dict[str, Any]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    return {}


def save_cache(cache: Dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def simple_correctness_check(answer: str, expected_answer: str) -> bool:
    """
    Simple keyword-overlap based correctness check.
    This is not perfect, but good enough for initial lab evaluation.
    Manual review is still recommended.
    """
    answer_lower = answer.lower()
    expected_lower = expected_answer.lower()

    important_words = []

    for word in expected_lower.replace(",", " ").replace(".", " ").split():
        word = word.strip()

        if len(word) <= 3:
            continue

        important_words.append(word)

    if not important_words:
        return False

    matched = 0

    for word in important_words:
        if word in answer_lower:
            matched += 1

    score = matched / len(important_words)

    return score >= 0.5


def run_single_benchmark_case(case: Dict[str, Any], cache: Dict[str, Any]) -> Dict[str, Any]:
    question_id = case["id"]
    question = case["question"]
    expected_answer = case["expected_answer"]
    question_type = case.get("type", "unknown")

    cache_key = str(question_id)

    if cache_key in cache:
        print(f"[CACHE] Question {question_id}: {question}")
        return cache[cache_key]

    print(f"\n[RUN] Question {question_id}: {question}")

    flat_start = time.perf_counter()
    flat_result = query_flat_rag(
        question=question,
        top_k=3,
        verbose=False
    )
    flat_time = time.perf_counter() - flat_start

    graph_start = time.perf_counter()
    graph_result = query_graph_rag(
        question=question,
        hops=2,
        verbose=False
    )
    graph_time = time.perf_counter() - graph_start

    flat_answer = flat_result["answer"]
    graph_answer = graph_result["answer"]

    flat_correct = simple_correctness_check(flat_answer, expected_answer)
    graph_correct = simple_correctness_check(graph_answer, expected_answer)

    result = {
        "id": question_id,
        "type": question_type,
        "question": question,
        "expected_answer": expected_answer,

        "flat_answer": flat_answer,
        "flat_correct_auto": flat_correct,
        "flat_time_seconds": round(flat_time, 4),
        "flat_context": flat_result.get("context", ""),

        "graph_entity": graph_result.get("entity"),
        "graph_answer": graph_answer,
        "graph_correct_auto": graph_correct,
        "graph_time_seconds": round(graph_time, 4),
        "graph_context": graph_result.get("graph_context", ""),

        "manual_note": ""
    }

    cache[cache_key] = result
    save_cache(cache)

    return result


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)

    flat_correct = sum(1 for item in results if item["flat_correct_auto"])
    graph_correct = sum(1 for item in results if item["graph_correct_auto"])

    avg_flat_time = sum(item["flat_time_seconds"] for item in results) / total
    avg_graph_time = sum(item["graph_time_seconds"] for item in results) / total

    summary = {
        "total_questions": total,
        "flat_correct_auto": flat_correct,
        "graph_correct_auto": graph_correct,
        "flat_accuracy_auto": round(flat_correct / total, 4),
        "graph_accuracy_auto": round(graph_correct / total, 4),
        "avg_flat_time_seconds": round(avg_flat_time, 4),
        "avg_graph_time_seconds": round(avg_graph_time, 4)
    }

    return summary


def save_results(results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "summary": summary,
        "results": results
    }

    OUTPUT_JSON_PATH.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    table_rows = []

    for item in results:
        table_rows.append({
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "flat_answer": item["flat_answer"],
            "flat_correct_auto": item["flat_correct_auto"],
            "flat_time_seconds": item["flat_time_seconds"],
            "graph_entity": item["graph_entity"],
            "graph_answer": item["graph_answer"],
            "graph_correct_auto": item["graph_correct_auto"],
            "graph_time_seconds": item["graph_time_seconds"],
            "manual_note": item["manual_note"]
        })

    df = pd.DataFrame(table_rows)
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"\nSaved JSON results to: {OUTPUT_JSON_PATH}")
    print(f"Saved CSV results to: {OUTPUT_CSV_PATH}")


def main():
    questions = load_benchmark_questions()
    cache = load_cache()

    print("Benchmark Flat RAG vs GraphRAG")
    print("==============================")
    print(f"Total questions: {len(questions)}")

    results = []

    for case in questions:
        result = run_single_benchmark_case(case, cache)
        results.append(result)

    summary = summarize_results(results)
    save_results(results, summary)

    print("\nBenchmark Summary")
    print("=================")
    print(f"Total questions: {summary['total_questions']}")
    print(f"Flat RAG correct: {summary['flat_correct_auto']}/{summary['total_questions']}")
    print(f"GraphRAG correct: {summary['graph_correct_auto']}/{summary['total_questions']}")
    print(f"Flat RAG accuracy: {summary['flat_accuracy_auto']}")
    print(f"GraphRAG accuracy: {summary['graph_accuracy_auto']}")
    print(f"Avg Flat RAG time: {summary['avg_flat_time_seconds']}s")
    print(f"Avg GraphRAG time: {summary['avg_graph_time_seconds']}s")


if __name__ == "__main__":
    main()