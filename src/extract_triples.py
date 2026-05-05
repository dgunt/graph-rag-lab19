import os
import json
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DATA_PATH = Path("data/tech_company_corpus.txt")
OUTPUT_PATH = Path("outputs/triples.json")


def load_corpus() -> str:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Corpus file not found: {DATA_PATH}")

    return DATA_PATH.read_text(encoding="utf-8")


def split_into_chunks(text: str) -> List[str]:
    """
    Split corpus by paragraphs.
    Each paragraph is treated as one chunk.
    """
    chunks = []

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph:
            chunks.append(paragraph)

    return chunks


def extract_triples_from_chunk(chunk: str) -> List[Dict[str, str]]:
    prompt = f"""
You are an information extraction system.

Extract knowledge graph triples from the text below.

Return ONLY valid JSON.
Do not include markdown.
Do not explain.

Each triple must have this format:
[
  {{
    "subject": "Entity A",
    "relation": "RELATION_TYPE",
    "object": "Entity B"
  }}
]

Rules:
- Use clear entity names.
- Use uppercase snake_case for relation.
- Extract company, founder, product, acquisition, investment, technology, and platform relations.
- Do not invent facts not present in the text.
- If no triples are found, return [].

Text:
{chunk}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You extract structured knowledge graph triples from text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    try:
        triples = json.loads(content)
    except json.JSONDecodeError:
        print("Failed to parse JSON from model output:")
        print(content)
        return []

    valid_triples = []

    for item in triples:
        if not isinstance(item, dict):
            continue

        subject = item.get("subject")
        relation = item.get("relation")
        obj = item.get("object")

        if subject and relation and obj:
            valid_triples.append({
                "subject": str(subject).strip(),
                "relation": str(relation).strip().upper(),
                "object": str(obj).strip()
            })

    return valid_triples


def deduplicate_triples(triples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    unique_triples = []

    for triple in triples:
        key = (
            triple["subject"].lower(),
            triple["relation"].upper(),
            triple["object"].lower()
        )

        if key not in seen:
            seen.add(key)
            unique_triples.append(triple)

    return unique_triples


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus()
    chunks = split_into_chunks(corpus)

    print(f"Loaded corpus.")
    print(f"Total chunks: {len(chunks)}")
    print(f"Using model: {MODEL}")

    all_triples = []

    for chunk in tqdm(chunks, desc="Extracting triples"):
        triples = extract_triples_from_chunk(chunk)
        all_triples.extend(triples)

    all_triples = deduplicate_triples(all_triples)

    OUTPUT_PATH.write_text(
        json.dumps(all_triples, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Done.")
    print(f"Total triples: {len(all_triples)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()