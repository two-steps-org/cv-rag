"""Embedding service - generates vector embeddings using Voyage AI.

Uses the voyage-3 model (1024-dimension vectors) with asymmetric search
support: documents are embedded with input_type="document" and queries
with input_type="query".
"""

import voyageai

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

MODEL = "voyage-3"
BATCH_SIZE = 128  # VoyageAI max texts per request

_client: voyageai.AsyncClient | None = None


def _get_client() -> voyageai.AsyncClient:
    """Return the shared AsyncClient, creating it on first call."""
    global _client
    if _client is None:
        _client = voyageai.AsyncClient(api_key=settings.voyage_api_key)
        logger.info("VoyageAI AsyncClient initialized (model=%s)", MODEL)
    return _client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of document texts into float vectors.

    Handles batching automatically -- VoyageAI accepts at most 128 texts
    per request, so larger lists are split into sequential batches.

    Args:
        texts: Plain-text strings to embed (resume chunks, etc.).

    Returns:
        List of 1024-dimension float vectors, one per input text.

    Raises:
        voyageai.error.VoyageError: On API-level failures after retries.
        ValueError: If *texts* is empty.
    """
    if not texts:
        raise ValueError("embed_texts requires at least one text")

    client = _get_client()
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch = texts[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1

        try:
            result = await client.embed(
                batch,
                model=MODEL,
                input_type="document",
            )
            all_embeddings.extend(result.embeddings)
            logger.debug(
                "Embedded batch %d/%d (%d texts, %d tokens)",
                batch_num,
                total_batches,
                len(batch),
                result.total_tokens,
            )
        except Exception:
            logger.exception(
                "VoyageAI embed failed on batch %d/%d (%d texts)",
                batch_num,
                total_batches,
                len(batch),
            )
            raise

    logger.info("Embedded %d texts in %d batch(es)", len(texts), total_batches)
    return all_embeddings


async def embed_query(text: str) -> list[float]:
    """Embed a single search query into a float vector.

    Uses ``input_type="query"`` so the resulting vector is optimised for
    asymmetric retrieval against document embeddings.

    Args:
        text: The search query string.

    Returns:
        A single 1024-dimension float vector.

    Raises:
        voyageai.error.VoyageError: On API-level failures.
        ValueError: If *text* is empty or blank.
    """
    if not text or not text.strip():
        raise ValueError("embed_query requires a non-empty query string")

    client = _get_client()

    try:
        result = await client.embed(
            [text],
            model=MODEL,
            input_type="query",
        )
        logger.debug("Embedded query (%d tokens)", result.total_tokens)
        return result.embeddings[0]
    except Exception:
        logger.exception("VoyageAI embed_query failed")
        raise
