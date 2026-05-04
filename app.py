from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent.router import DEFAULT_CONTEXT
from agent.core import run_agent
from agent.session_store import (
    delete_session,
    get_session_messages,
    list_session_ids,
    reset_session,
)


app = FastAPI(title="Minimal Agent API")


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


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok"
    }


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id.strip() or "default"
    context = DEFAULT_CONTEXT.copy()

    if request.context:
        context.update(request.context)

    if request.reset:
        messages = reset_session(session_id)
    else:
        messages = get_session_messages(session_id)

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
