"""Vector store service - manages MongoDB Atlas vector search operations."""

from app.database import VECTOR_INDEX_NAME, get_db
from app.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "resume_chunks"
EMBEDDING_PATH = "embedding"
MIN_NUM_CANDIDATES = 100


async def store_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
) -> int:
    """Persist chunked resume data with their vector embeddings.

    Merges each embedding into its corresponding chunk dict and inserts
    all documents into the resume_chunks collection in a single batch.

    Args:
        chunks: List of chunk dicts produced by chunker.chunk_resume().
        embeddings: List of embedding vectors aligned by index with chunks.

    Returns:
        Number of documents inserted.

    Raises:
        ValueError: If chunks and embeddings lists have different lengths.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
        )

    if not chunks:
        logger.warning("store_chunks called with an empty list; nothing to insert")
        return 0

    documents = [{**chunk, EMBEDDING_PATH: embedding} for chunk, embedding in zip(chunks, embeddings)]

    db = get_db()
    result = await db[COLLECTION_NAME].insert_many(documents)
    inserted_count = len(result.inserted_ids)

    logger.info("Inserted %d chunks into %s", inserted_count, COLLECTION_NAME)
    return inserted_count


async def search_similar(
    query_embedding: list[float],
    top_k: int = 5,
    position_tag: str | None = None,
    candidate_name: str | None = None,
) -> list[dict]:
    """Find the most similar resume chunks using MongoDB Atlas vector search.

    Builds a $vectorSearch aggregation pipeline with optional pre-filters
    on position_tag and candidate_name, then projects the relevant fields
    along with the similarity score.

    Args:
        query_embedding: The query vector to search against.
        top_k: Maximum number of results to return.
        position_tag: Optional filter to restrict results to a specific position.
        candidate_name: Optional filter to restrict results to a specific candidate.

    Returns:
        List of result dicts, each containing text, metadata, and a score field.
    """
    num_candidates = max(top_k * 10, MIN_NUM_CANDIDATES)

    # Build the $vectorSearch stage
    vector_search_stage: dict = {
        "$vectorSearch": {
            "index": VECTOR_INDEX_NAME,
            "path": EMBEDDING_PATH,
            "queryVector": query_embedding,
            "numCandidates": num_candidates,
            "limit": top_k,
        }
    }

    # Add pre-filter if any filter criteria are provided
    pre_filter: dict = {}
    if position_tag:
        pre_filter["position_tag"] = position_tag
    if candidate_name:
        pre_filter["candidate_name"] = candidate_name

    if pre_filter:
        vector_search_stage["$vectorSearch"]["filter"] = pre_filter

    # Project relevant fields and the vector search score
    project_stage = {
        "$project": {
            "_id": 0,
            "text": 1,
            "candidate_name": 1,
            "section_type": 1,
            "file_name": 1,
            "position_tag": 1,
            "resume_id": 1,
            "score": {"$meta": "vectorSearchScore"},
        }
    }

    pipeline = [vector_search_stage, project_stage]

    db = get_db()
    results = await db[COLLECTION_NAME].aggregate(pipeline).to_list(length=top_k)

    logger.info(
        "Vector search returned %d results (top_k=%d, filters=%s)",
        len(results),
        top_k,
        pre_filter or "none",
    )
    return results


async def delete_by_resume_id(resume_id: str) -> int:
    """Delete all chunks associated with a given resume.

    Args:
        resume_id: The resume identifier whose chunks should be removed.

    Returns:
        Number of documents deleted.
    """
    db = get_db()
    result = await db[COLLECTION_NAME].delete_many({"resume_id": resume_id})

    logger.info(
        "Deleted %d chunks for resume_id='%s'",
        result.deleted_count,
        resume_id,
    )
    return result.deleted_count


async def get_all_chunks_for_resume(resume_id: str) -> list[dict]:
    """Retrieve all chunks for a resume without embedding vectors.

    Useful for reconstructing the full resume content or inspecting
    stored chunks without the overhead of returning large embedding arrays.

    Args:
        resume_id: The resume identifier to look up.

    Returns:
        List of chunk dicts sorted by chunk_index, excluding the embedding field.
    """
    db = get_db()
    cursor = db[COLLECTION_NAME].find(
        {"resume_id": resume_id},
        {"_id": 0, EMBEDDING_PATH: 0},
    ).sort("chunk_index", 1)

    chunks = await cursor.to_list(length=None)

    logger.info(
        "Retrieved %d chunks for resume_id='%s'",
        len(chunks),
        resume_id,
    )
    return chunks
