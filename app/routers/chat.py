"""Chat router - handles chat/query interactions with the HR agent.

Provides SSE streaming chat, session listing, and session deletion.
"""

import json
from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.agent.graph import stream_agent
from app.database import get_db
from app.logger import get_logger
from app.models.schemas import ChatRequest, ChatSessionItem

logger = get_logger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _sse_event(payload: dict) -> str:
    """Format a dict as a Server-Sent Event data line."""
    return f"data: {json.dumps(payload)}\n\n"


def _parse_session_id(session_id: str) -> ObjectId:
    """Parse a string into a BSON ObjectId, raising HTTP 400 on invalid format."""
    try:
        return ObjectId(session_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid session_id")


async def _get_or_create_session(
    session_id: str | None,
    position_tag: str | None,
) -> tuple[ObjectId, list[dict], str | None]:
    """Load an existing session or create a new one.

    Returns:
        Tuple of (session ObjectId, messages list, position_tag from session).
    """
    db = get_db()
    collection = db["chat_sessions"]

    if session_id is not None:
        oid = _parse_session_id(session_id)

        session = await collection.find_one({"_id": oid})
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        return oid, session.get("messages") or [], session.get("position_tag")

    now = datetime.now(timezone.utc)
    result = await collection.insert_one(
        {
            "messages": [],
            "position_tag": position_tag,
            "created_at": now,
            "updated_at": now,
        }
    )
    return result.inserted_id, [], position_tag


async def _event_generator(
    request: ChatRequest,
    session_oid: ObjectId,
    history: list[dict],
    position_tag: str | None,
    model: str | None,
):
    """Async generator that yields SSE-formatted strings.

    Streams tokens from the agent graph, then persists the full
    conversation turn to MongoDB once streaming completes.
    """
    session_id_str = str(session_oid)

    yield _sse_event({"type": "session", "session_id": session_id_str})

    collected_tokens: list[str] = []

    try:
        async for event in stream_agent(
            message=request.message,
            position_tag=position_tag or "",
            history=history or [],
            model=model,
        ):
            event_type = event.get("type")

            if event_type == "token":
                content = event.get("content", "")
                collected_tokens.append(content)
                yield _sse_event({"type": "token", "content": content})

            elif event_type == "tool_call":
                yield _sse_event({
                    "type": "tool_call",
                    "name": event.get("name", ""),
                    "args": event.get("args", {}),
                })

            elif event_type == "error":
                yield _sse_event({"type": "error", "content": event.get("content", "")})

            elif event_type == "done":
                yield _sse_event({"type": "done", "session_id": session_id_str})

    except Exception as exc:
        logger.error("SSE stream error for session %s: %s", session_id_str, exc)
        yield _sse_event({"type": "error", "content": str(exc)})
        yield _sse_event({"type": "done", "session_id": session_id_str})

    # Persist the conversation turn to MongoDB
    full_response = "".join(collected_tokens)
    try:
        db = get_db()
        await db["chat_sessions"].update_one(
            {"_id": session_oid},
            {
                "$push": {
                    "messages": {
                        "$each": [
                            {"role": "user", "content": request.message},
                            {"role": "assistant", "content": full_response},
                        ]
                    }
                },
                "$set": {"updated_at": datetime.now(timezone.utc)},
            },
        )
        logger.info(
            "Persisted conversation turn to session %s (%d token chunks)",
            session_id_str,
            len(collected_tokens),
        )
    except Exception as exc:
        logger.error(
            "Failed to persist conversation to session %s: %s",
            session_id_str,
            exc,
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("")
async def chat(request: ChatRequest):
    """SSE streaming chat endpoint.

    Accepts a user message, streams the agent response as Server-Sent Events,
    and persists the full conversation turn to the session in MongoDB.
    """
    session_oid, history, position_tag = await _get_or_create_session(
        request.session_id,
        request.position_tag,
    )

    model = request.model
    logger.info(
        "Chat request received for session %s using model: %s",
        session_oid,
        model or "claude-sonnet-4-5-20250929 (default)",
    )

    return StreamingResponse(
        _event_generator(request, session_oid, history, position_tag, model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions", response_model=list[ChatSessionItem])
async def list_sessions():
    """List all chat sessions, sorted by most recently updated."""
    db = get_db()
    cursor = db["chat_sessions"].find().sort("updated_at", -1)

    sessions: list[ChatSessionItem] = []
    async for doc in cursor:
        sessions.append(
            ChatSessionItem(
                id=str(doc["_id"]),
                created_at=doc["created_at"],
                updated_at=doc["updated_at"],
                message_count=len(doc.get("messages", [])),
                position_tag=doc.get("position_tag"),
            )
        )

    return sessions


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session by ID."""
    db = get_db()
    oid = _parse_session_id(session_id)

    result = await db["chat_sessions"].delete_one({"_id": oid})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"deleted": True}
