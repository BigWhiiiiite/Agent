import json


def print_messages(title: str, messages: list) -> None:
    print(f"\n{title}")
    for index, message in enumerate(messages, start=1):
        print(f"{index}. role={message['role']}")
        print(json.dumps(message, ensure_ascii=False, indent=2))


def print_tool_trace(tool_call: dict, tool_message: dict) -> None:
    function_info = tool_call.get("function", {})
    function_name = function_info.get("name")
    raw_arguments = function_info.get("arguments", {})
    payload = json.loads(tool_message["content"])
    status = "成功" if payload["ok"] else "失败"

    print("\n工具调用 trace")
    print(f"tool={function_name}")
    print(f"arguments={raw_arguments}")
    print(f"status={status}")

    if not payload["ok"]:
        print(f"error={payload['error']['message']}")
