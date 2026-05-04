from openai import OpenAI

from agent.config import MODEL_NAME


def build_completion_kwargs(messages: list, tools: list) -> dict:
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
    }

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    return kwargs


def call_llm(messages: list, tools: list) -> dict:
    client = OpenAI()
    response = client.chat.completions.create(
        **build_completion_kwargs(messages, tools)
    )

    assistant_message = response.choices[0].message
    return assistant_message.model_dump(exclude_none=True)


def append_tool_call_delta(tool_calls_by_index: dict, tool_call_delta: dict) -> None:
    index = tool_call_delta["index"]
    tool_call = tool_calls_by_index.setdefault(
        index,
        {
            "type": "function",
            "function": {
                "arguments": ""
            }
        }
    )

    if "id" in tool_call_delta:
        tool_call["id"] = tool_call_delta["id"]

    if "type" in tool_call_delta:
        tool_call["type"] = tool_call_delta["type"]

    function_delta = tool_call_delta.get("function", {})

    if "name" in function_delta:
        tool_call["function"]["name"] = function_delta["name"]

    if "arguments" in function_delta:
        tool_call["function"]["arguments"] += function_delta["arguments"]


def build_streamed_assistant_message(
    role: str,
    content_parts: list[str],
    tool_calls_by_index: dict
) -> dict:
    assistant_message = {
        "role": role
    }
    content = "".join(content_parts)

    if content:
        assistant_message["content"] = content

    if tool_calls_by_index:
        assistant_message["tool_calls"] = [
            tool_calls_by_index[index]
            for index in sorted(tool_calls_by_index)
        ]

    return assistant_message


def stream_llm(messages: list, tools: list):
    client = OpenAI()
    stream = client.chat.completions.create(
        **build_completion_kwargs(messages, tools),
        stream=True
    )
    role = "assistant"
    content_parts = []
    tool_calls_by_index = {}

    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        delta_dict = delta.model_dump(exclude_none=True)

        if "role" in delta_dict:
            role = delta_dict["role"]

        if "content" in delta_dict:
            content_delta = delta_dict["content"]
            content_parts.append(content_delta)
            yield {
                "type": "content_delta",
                "delta": content_delta
            }

        for tool_call_delta in delta_dict.get("tool_calls", []):
            append_tool_call_delta(tool_calls_by_index, tool_call_delta)

    yield {
        "type": "assistant_message",
        "message": build_streamed_assistant_message(
            role,
            content_parts,
            tool_calls_by_index
        )
    }
