from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent.core import create_initial_messages, run_agent
from agent.router import DEFAULT_CONTEXT


app = FastAPI(title="Minimal Agent API")


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户输入的问题")
    context: dict | None = Field(
        default=None,
        description="业务上下文，比如 user_role 和 page"
    )


class ChatResponse(BaseModel):
    answer: str
    context: dict


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok"
    }


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    context = DEFAULT_CONTEXT.copy()

    if request.context:
        context.update(request.context)

    messages = create_initial_messages()
    answer = run_agent(
        messages,
        request.message,
        debug=False,
        context=context
    )

    return ChatResponse(
        answer=answer,
        context=context
    )
