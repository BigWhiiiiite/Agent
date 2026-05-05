import json

from openai import OpenAI

from agent.config import MODEL_NAME


def format_message_for_memory(message: dict) -> str:
    role = message.get("role", "")
    content = message.get("content")

    if content:
        return f"{role}: {content}"

    tool_calls = message.get("tool_calls", [])

    if tool_calls:
        tool_names = [
            tool_call.get("function", {}).get("name", "")
            for tool_call in tool_calls
        ]
        tool_names = [
            tool_name
            for tool_name in tool_names
            if tool_name
        ]
        return f"{role}: 调用了工具 {', '.join(tool_names)}"

    return f"{role}:"


def format_turns_for_memory(turns: list[list]) -> str:
    lines = []

    for index, turn in enumerate(turns, start=1):
        lines.append(f"第 {index} 轮：")

        for message in turn:
            lines.append(format_message_for_memory(message))

    return "\n".join(lines)


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


def choose_tool_names_with_llm(
    user_input: str,
    context: dict,
    tool_catalog: list[dict]
) -> list[str]:
    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个 Agent 的工具选择器。"
                    "你只能根据工具名和一句话说明，判断当前问题可能需要哪些工具。"
                    "如果用户问题不需要工具，返回空数组。"
                    "不要编造工具名。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{user_input}\n"
                    f"当前上下文：{json.dumps(context, ensure_ascii=False)}\n"
                    f"候选工具清单：{json.dumps(tool_catalog, ensure_ascii=False)}\n\n"
                    "请只输出 JSON，格式为："
                    "{\"tool_names\":[\"工具名1\",\"工具名2\"]}"
                )
            }
        ]
    )
    content = response.choices[0].message.content or "{}"
    payload = json.loads(content)
    tool_names = payload.get("tool_names", [])

    if not isinstance(tool_names, list):
        return []

    return [
        tool_name
        for tool_name in tool_names
        if isinstance(tool_name, str)
    ]


def summarize_memory_with_llm(
    previous_summary: str,
    removed_turns: list[list],
    max_chars: int
) -> str:
    transcript = format_turns_for_memory(removed_turns)
    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个 Agent 的对话记忆整理器。"
                    "你的任务是把较早的对话压缩成结构化摘要，"
                    "帮助后续 LLM 理解用户当前问题的上下文。"
                    "只保留对未来回答可能有用的信息：用户需求、偏好、已确认事实、"
                    "重要实体、工具查询结果、未完成事项。"
                    "不要编造原对话里没有的信息。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"已有摘要：\n{previous_summary or '无'}\n\n"
                    f"需要压缩的旧对话：\n{transcript}\n\n"
                    f"请输出不超过 {max_chars} 字的中文摘要，固定使用下面结构：\n"
                    "用户目标/问题：\n"
                    "已知事实：\n"
                    "用户偏好：\n"
                    "工具结果：\n"
                    "未完成事项：\n"
                    "没有的信息写“无”。不要输出额外标题。"
                )
            }
        ]
    )

    return response.choices[0].message.content or ""


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
