"""Database module - MongoDB connection management and index setup."""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.operations import SearchIndexModel

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

VECTOR_INDEX_NAME = "resume_vector_index"

client: AsyncIOMotorClient | None = None
db: AsyncIOMotorDatabase | None = None


async def connect_db() -> None:
    global client, db
    client = AsyncIOMotorClient(settings.atlas_connection_string)
    db = client[settings.database_name]
    await client.admin.command("ping")


async def close_db() -> None:
    global client
    if client:
        client.close()


async def ensure_vector_index() -> None:
    """Create the Atlas Vector Search index on resume_chunks if it does not exist."""
    collection = get_db()["resume_chunks"]

    index_def = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1024,
                    "similarity": "cosine",
                },
                {
                    "type": "filter",
                    "path": "position_tag",
                },
                {
                    "type": "filter",
                    "path": "candidate_name",
                },
            ]
        },
        name=VECTOR_INDEX_NAME,
        type="vectorSearch",
    )

    try:
        await collection.create_search_index(model=index_def)
        logger.info("Created vector search index '%s'.", VECTOR_INDEX_NAME)
    except Exception as exc:
        msg = str(exc)
        if "already exists" in msg.lower() or "duplicate" in msg.lower():
            logger.info("Vector search index '%s' already exists.", VECTOR_INDEX_NAME)
        else:
            logger.error("Failed to create vector search index: %s", msg)
            raise


def get_db() -> AsyncIOMotorDatabase:
    if db is None:
        raise RuntimeError("Database not initialized. Call connect_db() first.")
    return db
