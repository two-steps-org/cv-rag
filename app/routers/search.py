"""Search router - debug endpoint for vector similarity search against resume chunks."""

from fastapi import APIRouter, HTTPException

from app.logger import get_logger
from app.models.schemas import SearchRequest, SearchResult
from app.services.embeddings import embed_query
from app.services.vector_store import search_similar

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=list[SearchResult])
async def search_resumes(body: SearchRequest) -> list[SearchResult]:
    """Run a vector similarity search against stored resume chunks.
 
    Embeds the incoming query text via VoyageAI, then performs a
    MongoDB Atlas vector search to find the closest matching chunks.
    Intended as a debug/testing endpoint for the embedding pipeline.
    """
    try:
        query_embedding = await embed_query(body.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Failed to embed search query")
        raise HTTPException(status_code=502, detail="Embedding service unavailable")

    try:
        raw_results = await search_similar(
            query_embedding=query_embedding,
            top_k=body.top_k,
            position_tag=body.position_tag,
        )
    except Exception:
        logger.exception("Vector search failed")
        raise HTTPException(status_code=502, detail="Vector search service unavailable")

    results = [
        SearchResult(
            text=doc.get("text", ""),
            candidate_name=doc.get("candidate_name", ""),
            section_type=doc.get("section_type", ""),
            file_name=doc.get("file_name", ""),
            position_tag=doc.get("position_tag"),
            resume_id=doc.get("resume_id", ""),
            score=doc.get("score", 0.0),
        )
        for doc in raw_results
    ]

    logger.info(
        "Search returned %d results for query='%s' (top_k=%d, position_tag=%s)",
        len(results),
        body.query[:80],
        body.top_k,
        body.position_tag,
    )
    return results
