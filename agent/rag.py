import json
import math

from openai import OpenAI

from agent.config import (
    DOCS_DIR,
    EMBEDDING_CACHE_PATH,
    EMBEDDING_MODEL,
    MIN_SEMANTIC_SIMILARITY,
    TOP_K,
    VECTOR_INDEX_PATH,
)


def load_embedding_cache() -> dict:
    if not EMBEDDING_CACHE_PATH.exists():
        return {}

    return json.loads(EMBEDDING_CACHE_PATH.read_text(encoding="utf-8"))


EMBEDDING_CACHE = load_embedding_cache()


def save_embedding_cache() -> None:
    EMBEDDING_CACHE_PATH.write_text(
        json.dumps(EMBEDDING_CACHE, ensure_ascii=False),
        encoding="utf-8"
    )


def load_vector_index() -> dict:
    if not VECTOR_INDEX_PATH.exists():
        return {}

    return json.loads(VECTOR_INDEX_PATH.read_text(encoding="utf-8"))


def save_vector_index(index: dict) -> None:
    VECTOR_INDEX_PATH.write_text(
        json.dumps(index, ensure_ascii=False),
        encoding="utf-8"
    )


def build_search_terms(query: str) -> set:
    cleaned_query = query
    for char in " ，。？！?：:、\n\t":
        cleaned_query = cleaned_query.replace(char, "")

    terms = set()

    if cleaned_query:
        terms.add(cleaned_query)

    for index in range(len(cleaned_query) - 1):
        terms.add(cleaned_query[index:index + 2])

    return terms


def load_knowledge_chunks() -> list:
    chunks = []

    for path in DOCS_DIR.glob("*.txt"):
        content = path.read_text(encoding="utf-8")
        paragraphs = [
            paragraph.strip()
            for paragraph in content.split("\n\n")
            if paragraph.strip()
        ]

        for index, paragraph in enumerate(paragraphs, start=1):
            chunks.append({
                "source": path.name,
                "chunk_id": f"{path.stem}-{index}",
                "content": paragraph
            })

    return chunks


def score_chunk(query: str, terms: set, chunk: dict) -> int:
    content = chunk["content"]
    score = sum(1 for term in terms if term in content)

    if query in content:
        score += 5

    return score


def search_knowledge_base(query: str) -> dict:
    terms = build_search_terms(query)
    results = []

    for chunk in load_knowledge_chunks():
        score = score_chunk(query, terms, chunk)

        if score > 0:
            results.append({
                **chunk,
                "score": score
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    top_results = results[:TOP_K]

    return {
        "query": query,
        "found": bool(top_results),
        "result_count": len(top_results),
        "top_k": TOP_K,
        "results": top_results
    }


def get_embedding(text: str) -> list:
    if text in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[text]

    client = OpenAI()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        encoding_format="float"
    )
    embedding = response.data[0].embedding
    EMBEDDING_CACHE[text] = embedding
    save_embedding_cache()
    return embedding


def cosine_similarity(vector_a: list, vector_b: list) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def build_vector_index() -> dict:
    chunks = []

    for chunk in load_knowledge_chunks():
        chunks.append({
            **chunk,
            "embedding": get_embedding(chunk["content"])
        })

    index = {
        "embedding_model": EMBEDDING_MODEL,
        "chunks": chunks
    }
    save_vector_index(index)
    return index


def get_vector_index() -> dict:
    index = load_vector_index()
    current_chunks = load_knowledge_chunks()
    indexed_chunks = index.get("chunks", [])

    current_signature = [
        (chunk["source"], chunk["chunk_id"], chunk["content"])
        for chunk in current_chunks
    ]
    indexed_signature = [
        (chunk["source"], chunk["chunk_id"], chunk["content"])
        for chunk in indexed_chunks
    ]

    if (
        index.get("embedding_model") != EMBEDDING_MODEL
        or current_signature != indexed_signature
    ):
        return build_vector_index()

    return index


def semantic_search_knowledge_base(query: str) -> dict:
    query_embedding = get_embedding(query)
    vector_index = get_vector_index()
    results = []

    for chunk in vector_index["chunks"]:
        chunk_embedding = chunk["embedding"]
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        results.append({
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"],
            "similarity": round(similarity, 4)
        })

    results.sort(key=lambda item: item["similarity"], reverse=True)
    top_results = [
        result for result in results
        if result["similarity"] >= MIN_SEMANTIC_SIMILARITY
    ][:TOP_K]

    return {
        "query": query,
        "found": bool(top_results),
        "result_count": len(top_results),
        "embedding_model": EMBEDDING_MODEL,
        "index_path": VECTOR_INDEX_PATH.name,
        "min_similarity": MIN_SEMANTIC_SIMILARITY,
        "top_k": TOP_K,
        "results": top_results
    }
