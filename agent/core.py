from agent.config import MAX_AGENT_STEPS
from agent.debug import print_messages, print_tool_trace
from agent.executor import execute_tool_call
from agent.llm import call_llm, stream_llm
from agent.memory import trim_messages
from agent.router import DEFAULT_CONTEXT, select_tools
from agent.tracing import append_tool_call_trace, create_trace, finish_trace


def create_initial_messages() -> list:
    return [
        {
            "role": "system",
            "content": (
                "你是一个校园助手。"
                "你可以查询老师空闲时间、课程信息，也可以搜索本地知识库。"
                "如果用户的问题需要工具，并且参数已经足够，就调用最合适的工具。"
                "回答学校制度、课程规则、Agent或RAG概念时，优先搜索本地知识库。"
                "如果用户的问题和知识库里的原文措辞可能不一致，优先使用语义检索工具。"
                "如果缺少必要参数，就先追问用户，不要编造参数。"
                "回答知识库问题时，要基于工具返回的 results，并可简短提及来源 source。"
                "如果知识库工具返回 found=false，就如实说明没有查到相关资料。"
                "工具结果里的 ok=false 表示工具失败，需要根据 error.message 向用户简短说明。"
                "最终回答要简短、自然。"
            )
        }
    ]


def run_agent(
    messages: list,
    user_input: str,
    debug: bool = False,
    context: dict | None = None
) -> str:
    context = context or DEFAULT_CONTEXT
    available_tools = select_tools(user_input, context)
    selected_tool_names = [tool["function"]["name"] for tool in available_tools]
    trace = create_trace(user_input, context, selected_tool_names)

    messages.append({
        "role": "user",
        "content": user_input
    })
    trimmed_count = trim_messages(messages)

    if debug:
        if trimmed_count:
            print(f"\n已裁剪较早的 {trimmed_count} 条 messages")
        print_messages("用户输入后 messages", messages)
        print(f"\n本轮候选工具：{selected_tool_names}")

    for _ in range(MAX_AGENT_STEPS):
        trimmed_count = trim_messages(messages)
        if debug and trimmed_count:
            print(f"\n已裁剪较早的 {trimmed_count} 条 messages")

        assistant_message = call_llm(messages, tools=available_tools)
        messages.append(assistant_message)

        if debug:
            print_messages("模型返回 assistant message 后", messages)

        if "tool_calls" not in assistant_message:
            final_answer = assistant_message["content"]
            trim_messages(messages)
            finish_trace(trace, final_answer, "final_answer")
            return final_answer

        for tool_call in assistant_message["tool_calls"]:
            tool_message = execute_tool_call(tool_call)
            messages.append(tool_message)
            append_tool_call_trace(trace, tool_call, tool_message)

            if debug:
                print_tool_trace(tool_call, tool_message)

        if debug:
            print_messages("工具执行完成并加入 tool message 后", messages)

    final_answer = "工具调用轮数超过上限，已停止。请换一种问法或稍后再试。"
    trim_messages(messages)
    finish_trace(trace, final_answer, "max_steps")
    return final_answer


def run_agent_stream(
    messages: list,
    user_input: str,
    debug: bool = False,
    context: dict | None = None
):
    context = context or DEFAULT_CONTEXT
    available_tools = select_tools(user_input, context)
    selected_tool_names = [tool["function"]["name"] for tool in available_tools]
    trace = create_trace(user_input, context, selected_tool_names)

    messages.append({
        "role": "user",
        "content": user_input
    })
    trimmed_count = trim_messages(messages)

    if debug:
        if trimmed_count:
            print(f"\n已裁剪较早的 {trimmed_count} 条 messages")
        print_messages("用户输入后 messages", messages)
        print(f"\n本轮候选工具：{selected_tool_names}")

    for _ in range(MAX_AGENT_STEPS):
        trimmed_count = trim_messages(messages)
        if debug and trimmed_count:
            print(f"\n已裁剪较早的 {trimmed_count} 条 messages")

        assistant_message = None

        for event in stream_llm(messages, tools=available_tools):
            if event["type"] == "content_delta":
                yield event["delta"]

            if event["type"] == "assistant_message":
                assistant_message = event["message"]

        if assistant_message is None:
            assistant_message = {
                "role": "assistant",
                "content": ""
            }

        messages.append(assistant_message)

        if debug:
            print_messages("模型返回 assistant message 后", messages)

        if "tool_calls" not in assistant_message:
            final_answer = assistant_message.get("content", "")
            trim_messages(messages)
            finish_trace(trace, final_answer, "final_answer")
            return

        for tool_call in assistant_message["tool_calls"]:
            tool_message = execute_tool_call(tool_call)
            messages.append(tool_message)
            append_tool_call_trace(trace, tool_call, tool_message)

            if debug:
                print_tool_trace(tool_call, tool_message)

        if debug:
            print_messages("工具执行完成并加入 tool message 后", messages)

    final_answer = "工具调用轮数超过上限，已停止。请换一种问法或稍后再试。"
    trim_messages(messages)
    finish_trace(trace, final_answer, "max_steps")

    for char in final_answer:
        yield char
