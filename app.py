import json

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.config import BASE_DIR
from agent.router import DEFAULT_CONTEXT
from agent.core import run_agent, run_agent_stream
from agent.session_store import (
    delete_session,
    list_session_ids,
    use_session_messages,
)


STATIC_DIR = BASE_DIR / "static"
app = FastAPI(title="Minimal Agent API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户输入的问题")
    session_id: str = Field(
        default="default",
        description="会话 ID。同一个 session_id 会复用同一份 messages。"
    )
    context: dict | None = Field(
        default=None,
        description="业务上下文，比如 user_role 和 page"
    )
    reset: bool = Field(
        default=False,
        description="是否在本次对话前重置这个会话"
    )


class ChatResponse(BaseModel):
    answer: str
    context: dict
    session_id: str
    message_count: int


class SessionsResponse(BaseModel):
    sessions: list[str]


class DeleteSessionResponse(BaseModel):
    session_id: str
    deleted: bool


def build_chat_context(request: ChatRequest) -> tuple[str, dict]:
    session_id = request.session_id.strip() or "default"
    context = DEFAULT_CONTEXT.copy()

    if request.context:
        context.update(request.context)

    return session_id, context


def format_sse_event(event: str, data: dict) -> str:
    json_data = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {json_data}\n\n"


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok"
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    session_id, context = build_chat_context(request)

    with use_session_messages(session_id, reset=request.reset) as messages:
        answer = run_agent(
            messages,
            request.message,
            debug=False,
            context=context
        )

        return ChatResponse(
            answer=answer,
            context=context,
            session_id=session_id,
            message_count=len(messages)
        )


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    session_id, context = build_chat_context(request)

    def event_generator():
        try:
            with use_session_messages(session_id, reset=request.reset) as messages:
                for delta in run_agent_stream(
                    messages,
                    request.message,
                    debug=False,
                    context=context
                ):
                    yield format_sse_event("delta", {"content": delta})

                yield format_sse_event(
                    "done",
                    {
                        "session_id": session_id,
                        "message_count": len(messages)
                    }
                )
        except Exception as error:
            yield format_sse_event(
                "error",
                {
                    "message": str(error)
                }
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache"
        }
    )


@app.get("/sessions")
def list_sessions() -> SessionsResponse:
    return SessionsResponse(
        sessions=list_session_ids()
    )


@app.delete("/sessions/{session_id}")
def remove_session(session_id: str) -> DeleteSessionResponse:
    deleted = delete_session(session_id)

    return DeleteSessionResponse(
        session_id=session_id,
        deleted=deleted
    )
