import json

from agent.tools import TOOL_REGISTRY


def build_tool_success(tool_name: str, data: dict) -> dict:
    return {
        "ok": True,
        "tool_name": tool_name,
        "data": data
    }


def build_tool_error(tool_name: str | None, error_type: str, message: str) -> dict:
    return {
        "ok": False,
        "tool_name": tool_name,
        "error": {
            "type": error_type,
            "message": message
        }
    }


def build_tool_message(tool_call_id: str, payload: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(payload, ensure_ascii=False)
    }


def parse_tool_arguments(arguments) -> dict:
    if isinstance(arguments, str):
        return json.loads(arguments)

    if isinstance(arguments, dict):
        return arguments

    raise TypeError("工具参数必须是 JSON 字符串或 dict。")


def execute_tool_call(tool_call: dict) -> dict:
    tool_call_id = tool_call.get("id", "unknown_tool_call")
    function_info = tool_call.get("function", {})
    function_name = function_info.get("name")
    raw_arguments = function_info.get("arguments", {})

    if function_name not in TOOL_REGISTRY:
        payload = build_tool_error(
            function_name,
            "unknown_tool",
            f"工具 {function_name} 没有注册，无法执行。"
        )
        return build_tool_message(tool_call_id, payload)

    try:
        arguments = parse_tool_arguments(raw_arguments)
    except (json.JSONDecodeError, TypeError) as error:
        payload = build_tool_error(
            function_name,
            "invalid_arguments",
            f"工具参数解析失败：{error}"
        )
        return build_tool_message(tool_call_id, payload)

    try:
        tool_function = TOOL_REGISTRY[function_name]
        result = tool_function(**arguments)
        payload = build_tool_success(function_name, result)
    except TypeError as error:
        payload = build_tool_error(
            function_name,
            "argument_mismatch",
            f"工具参数和函数签名不匹配：{error}"
        )
    except Exception as error:
        payload = build_tool_error(
            function_name,
            "tool_runtime_error",
            f"工具执行失败：{error}"
        )

    return build_tool_message(tool_call_id, payload)
