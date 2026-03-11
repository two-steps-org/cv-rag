"""Resume upload router - handles file uploads, listing, deletion, and download."""

from pathlib import Path

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.database import get_db
from app.logger import get_logger
from app.models.schemas import (
    ResumeDocument,
    ResumeListItem,
    ResumeUploadResponse,
    UploadBatchResponse,
)
from app.services.chunker import chunk_resume
from app.services.embeddings import embed_texts
from app.services.extractor import extract_candidate_info
from app.services.parser import parse_resume
from app.services.vector_store import store_chunks

logger = get_logger(__name__)

router = APIRouter(prefix="/api/resumes", tags=["resumes"])


def _parse_object_id(resume_id: str) -> ObjectId:
    """Parse a string into a BSON ObjectId, raising HTTP 400 on invalid format."""
    try:
        return ObjectId(resume_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid resume ID format")


async def _embed_and_store_chunks(chunks: list[dict], resume_id: str) -> None:
    """Generate embeddings for resume chunks and store them in the vector store.

    Runs as a background task after the upload response is returned.
    Updates the resume document's embedding_status to 'completed' on success
    or 'failed' on error.
    """
    db = get_db()
    try:
        texts = [chunk["text"] for chunk in chunks]
        embeddings = await embed_texts(texts)

        for chunk in chunks:
            chunk["resume_id"] = resume_id

        inserted = await store_chunks(chunks, embeddings)
        await db.resumes.update_one(
            {"_id": ObjectId(resume_id)},
            {"$set": {"embedding_status": "completed"}},
        )
        logger.info(
            "Background embedding completed for resume %s (%d chunks stored)",
            resume_id,
            inserted,
        )
    except Exception as exc:
        logger.error(
            "Background embedding failed for resume %s: %s", resume_id, exc
        )
        try:
            await db.resumes.update_one(
                {"_id": ObjectId(resume_id)},
                {"$set": {"embedding_status": "failed"}},
            )
        except Exception as update_exc:
            logger.error(
                "Failed to update embedding_status to 'failed' for resume %s: %s",
                resume_id,
                update_exc,
            )


@router.post("/upload", response_model=UploadBatchResponse)
async def upload_resumes(
    files: list[UploadFile],
    background_tasks: BackgroundTasks,
    position_tag: str = Form(None),
) -> UploadBatchResponse:
    """Upload one or more resume files (PDF or DOCX).

    Parses each file, extracts candidate info, chunks the text into
    sections, and stores both the document and its chunks in MongoDB.
    """
    db = get_db()
    uploaded: list[ResumeUploadResponse] = []
    errors: list[dict] = []

    uploads_path = Path(settings.uploads_dir)
    uploads_path.mkdir(parents=True, exist_ok=True)

    for file in files:
        filename = file.filename or "unknown"
        try:
            content = await file.read()
            if not content:
                raise ValueError("Empty file")

            raw_text = await parse_resume(content, filename)
            if not raw_text.strip():
                raise ValueError("No text could be extracted from the file")

            candidate_info = extract_candidate_info(raw_text)
            candidate_name = candidate_info["name"]

            chunks = chunk_resume(
                text=raw_text,
                candidate_name=candidate_name,
                file_name=filename,
                position_tag=position_tag,
            )

            embedding_status = "processing" if chunks else "pending"

            doc = ResumeDocument(
                candidate_name=candidate_name,
                file_name=filename,
                raw_text=raw_text,
                position_tag=position_tag,
                sections_count=len(chunks),
                embedding_status=embedding_status,
            )

            result = await db.resumes.insert_one(doc.model_dump())
            resume_id = result.inserted_id

            # Save original file to uploads directory
            file_path = uploads_path / f"{resume_id}_{filename}"
            file_path.write_bytes(content)

            await db.resumes.update_one(
                {"_id": resume_id},
                {"$set": {"file_path": str(file_path)}},
            )

            if chunks:
                background_tasks.add_task(
                    _embed_and_store_chunks, chunks, str(resume_id)
                )

            uploaded.append(
                ResumeUploadResponse(
                    id=str(resume_id),
                    candidate_name=candidate_name,
                    file_name=filename,
                    sections_count=len(chunks),
                    position_tag=position_tag,
                    embedding_status=embedding_status,
                )
            )
            logger.info(
                "Uploaded resume '%s' for '%s' (%d chunks)",
                filename,
                candidate_name,
                len(chunks),
            )

        except Exception as exc:
            logger.error("Failed to process file '%s': %s", filename, exc)
            errors.append({"file_name": filename, "error": str(exc)})

    return UploadBatchResponse(uploaded=uploaded, errors=errors)


@router.get("", response_model=list[ResumeListItem])
async def list_resumes(
    position_tag: str | None = None,
) -> list[ResumeListItem]:
    """List all uploaded resumes, optionally filtered by position tag."""
    db = get_db()

    query: dict = {}
    if position_tag:
        query["position_tag"] = position_tag

    cursor = db.resumes.find(query)
    items: list[ResumeListItem] = []

    async for doc in cursor:
        items.append(
            ResumeListItem(
                id=str(doc["_id"]),
                candidate_name=doc["candidate_name"],
                file_name=doc["file_name"],
                upload_date=doc["upload_date"],
                position_tag=doc.get("position_tag"),
                sections_count=doc.get("sections_count", 0),
            )
        )

    return items


@router.delete("/{resume_id}")
async def delete_resume(resume_id: str) -> dict:
    """Delete a resume and all its associated chunks, including the file on disk."""
    db = get_db()
    obj_id = _parse_object_id(resume_id)

    doc = await db.resumes.find_one({"_id": obj_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="Resume not found")

    await db.resumes.delete_one({"_id": obj_id})
    await db.resume_chunks.delete_many({"resume_id": resume_id})

    file_path = doc.get("file_path")
    if file_path:
        Path(file_path).unlink(missing_ok=True)

    logger.info("Deleted resume %s and its chunks", resume_id)
    return {"deleted": True}


# Content-type mapping for supported resume file extensions
_MEDIA_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


@router.get("/{resume_id}/download")
async def download_resume(resume_id: str) -> FileResponse:
    """Download the original uploaded resume file."""
    db = get_db()
    obj_id = _parse_object_id(resume_id)

    doc = await db.resumes.find_one({"_id": obj_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="Resume not found")

    file_path = doc.get("file_path")
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Resume file not found on disk")

    filename = doc.get("file_name", "resume")
    suffix = Path(filename).suffix.lower()
    media_type = _MEDIA_TYPES.get(suffix, "application/octet-stream")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type,
    )
