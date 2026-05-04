import json
from datetime import datetime, timezone

from agent.config import TRACE_LOG_PATH
from agent.executor import parse_tool_arguments


def create_trace(user_input: str, context: dict, selected_tools: list) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_input": user_input,
        "context": context,
        "selected_tools": selected_tools,
        "tool_calls": [],
        "final_answer": None,
        "stop_reason": None
    }


def summarize_tool_payload(payload: dict) -> dict:
    if not payload.get("ok"):
        return {
            "ok": False,
            "error": payload.get("error")
        }

    data = payload.get("data", {})

    if isinstance(data, dict) and "results" in data:
        return {
            "ok": True,
            "found": data.get("found"),
            "result_count": data.get("result_count"),
            "sources": [
                {
                    "source": result.get("source"),
                    "chunk_id": result.get("chunk_id")
                }
                for result in data.get("results", [])
            ]
        }

    return {
        "ok": True,
        "data": data
    }


def append_tool_call_trace(trace: dict, tool_call: dict, tool_message: dict) -> None:
    function_info = tool_call.get("function", {})
    raw_arguments = function_info.get("arguments", {})

    try:
        arguments = parse_tool_arguments(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        arguments = raw_arguments

    payload = json.loads(tool_message["content"])
    trace["tool_calls"].append({
        "tool_name": function_info.get("name"),
        "arguments": arguments,
        "result_summary": summarize_tool_payload(payload)
    })


def finish_trace(trace: dict, final_answer: str, stop_reason: str) -> None:
    trace["final_answer"] = final_answer
    trace["stop_reason"] = stop_reason
    write_trace(trace)


def write_trace(trace: dict) -> None:
    TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with TRACE_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(trace, ensure_ascii=False) + "\n")
