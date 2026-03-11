"""LangGraph agent tool definitions for the HR Resume Agent.

Provides three async LangChain tools that the agent can invoke:
    - search_resumes:       Vector-similarity search across resume chunks.
    - get_candidate_resume: Reconstruct the full resume for a named candidate.
    - list_candidates:      List all candidates in the system, optionally
                            filtered by position tag.

All tools are async because they call the async MongoDB / VoyageAI services.
"""

import re
from typing import Optional

from langchain_core.tools import tool

from app.database import get_db
from app.logger import get_logger
from app.services.embeddings import embed_query
from app.services.vector_store import get_all_chunks_for_resume, search_similar

logger = get_logger(__name__)


@tool
async def search_resumes(
    query: str,
    top_k: int = 10,
    position_tag: Optional[str] = None,
) -> str:
    """Search resume chunks by semantic similarity to a query.

    Embeds the query text using VoyageAI and performs a vector search
    against all stored resume chunks in MongoDB Atlas. Use this tool
    when you need to find candidates or resume sections relevant to
    specific skills, experiences, or qualifications.

    Args:
        query: Natural-language search query describing the desired
            skills, experience, or qualifications.
        top_k: Maximum number of matching chunks to return (default 10).
        position_tag: Optional position tag to narrow results to resumes
            uploaded under a specific job posting.

    Returns:
        A formatted string listing each matching chunk with candidate
        name, section type, relevance score, and a text snippet.
    """
    try:
        query_embedding = await embed_query(query)
        results = await search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            position_tag=position_tag,
        )
    except Exception as exc:
        logger.error("search_resumes tool failed: %s", exc)
        return f"Error performing resume search: {exc}"

    if not results:
        return "No matching resume chunks found for the given query."

    lines = [f"Found {len(results)} matching resume chunk(s):\n"]
    for i, chunk in enumerate(results, start=1):
        score = chunk.get("score", 0.0)
        candidate = chunk.get("candidate_name", "Unknown")
        section = chunk.get("section_type", "unknown")
        text = chunk.get("text", "")
        # Truncate long text snippets for readability
        snippet = text[:500] + "..." if len(text) > 500 else text

        lines.append(
            f"--- Result {i} ---\n"
            f"Candidate: {candidate}\n"
            f"Section:   {section}\n"
            f"Score:     {score:.4f}\n"
            f"Text:\n{snippet}\n"
        )

    return "\n".join(lines)


@tool
async def get_candidate_resume(candidate_name: str) -> str:
    """Retrieve and reconstruct the full resume for a specific candidate.

    Looks up the candidate in the resumes collection by name, then
    fetches all stored chunks for that resume and reassembles them
    organised by section type. Use this tool when you need to review
    a specific candidate's complete resume.

    Args:
        candidate_name: The exact name of the candidate whose resume
            should be retrieved.

    Returns:
        The reconstructed resume text organised by section, or an error
        message if the candidate is not found.
    """
    db = get_db()

    try:
        # Case-insensitive lookup to be forgiving of minor casing differences
        escaped_name = re.escape(candidate_name)
        resume_doc = await db.resumes.find_one(
            {"candidate_name": {"$regex": f"^{escaped_name}$", "$options": "i"}},
        )
    except Exception as exc:
        logger.error("get_candidate_resume DB lookup failed: %s", exc)
        return f"Error looking up candidate: {exc}"

    if resume_doc is None:
        return (
            f"No resume found for candidate '{candidate_name}'. "
            "Please check the name spelling or use the list_candidates tool "
            "to see available candidates."
        )

    resume_id = str(resume_doc["_id"])
    file_name = resume_doc.get("file_name", "unknown")
    position_tag = resume_doc.get("position_tag", "N/A")
    upload_date = resume_doc.get("upload_date", "N/A")

    try:
        chunks = await get_all_chunks_for_resume(resume_id)
    except Exception as exc:
        logger.error(
            "get_candidate_resume chunk retrieval failed for %s: %s",
            resume_id,
            exc,
        )
        return f"Error retrieving resume chunks: {exc}"

    if not chunks:
        return (
            f"Resume record exists for '{candidate_name}' but no chunks "
            "are stored yet (embedding may still be processing)."
        )

    # Group chunks by section type, preserving chunk_index order
    sections: dict[str, list[str]] = {}
    for chunk in chunks:
        section_type = chunk.get("section_type", "other")
        sections.setdefault(section_type, []).append(chunk.get("text", ""))

    # Build the output
    header = (
        f"Resume for: {candidate_name}\n"
        f"File:       {file_name}\n"
        f"Position:   {position_tag}\n"
        f"Uploaded:   {upload_date}\n"
        f"{'=' * 60}\n"
    )

    body_parts = []
    for section_type, texts in sections.items():
        section_title = section_type.replace("_", " ").title()
        body_parts.append(f"\n## {section_title}\n")
        body_parts.append("\n".join(texts))

    return header + "\n".join(body_parts)


@tool
async def list_candidates(position_tag: Optional[str] = None) -> str:
    """List all candidates in the system with their resume metadata.

    Queries the resumes collection and returns a formatted list of
    every candidate. Optionally filters by position tag. Use this
    tool to discover which candidates are available before performing
    deeper searches or retrieving individual resumes.

    Args:
        position_tag: Optional position tag to filter candidates by
            a specific job posting. If omitted, all candidates are
            returned.

    Returns:
        A formatted list of candidates with their file names, upload
        dates, and position tags, or a message if none are found.
    """
    db = get_db()

    query: dict = {}
    if position_tag:
        query["position_tag"] = position_tag

    try:
        cursor = db.resumes.find(
            query,
            {
                "_id": 1,
                "candidate_name": 1,
                "file_name": 1,
                "upload_date": 1,
                "position_tag": 1,
                "sections_count": 1,
                "embedding_status": 1,
            },
        )
        docs = await cursor.to_list(length=None)
    except Exception as exc:
        logger.error("list_candidates DB query failed: %s", exc)
        return f"Error listing candidates: {exc}"

    if not docs:
        filter_msg = f" for position '{position_tag}'" if position_tag else ""
        return f"No candidates found{filter_msg}."

    lines = [f"Found {len(docs)} candidate(s):\n"]
    for doc in docs:
        name = doc.get("candidate_name", "Unknown")
        file_name = doc.get("file_name", "N/A")
        upload_date = doc.get("upload_date", "N/A")
        pos_tag = doc.get("position_tag", "N/A")
        sections = doc.get("sections_count", 0)
        status = doc.get("embedding_status", "unknown")

        lines.append(
            f"- {name}\n"
            f"    File:       {file_name}\n"
            f"    Uploaded:   {upload_date}\n"
            f"    Position:   {pos_tag}\n"
            f"    Sections:   {sections}\n"
            f"    Embeddings: {status}\n"
        )

    return "\n".join(lines)


# Convenience list for registering all tools with the agent graph
agent_tools = [search_resumes, get_candidate_resume, list_candidates]
