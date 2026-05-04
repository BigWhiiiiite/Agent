import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import agent.core as agent_core
from agent.llm import append_tool_call_delta, build_streamed_assistant_message
from agent.memory import trim_messages
from agent.rag import search_knowledge_base
from agent.router import select_tool_names
from agent.tools import TOOL_REGISTRY


EVAL_CASES_PATH = ROOT_DIR / "eval_cases.json"


def load_eval_cases() -> list:
    return json.loads(EVAL_CASES_PATH.read_text(encoding="utf-8"))


def evaluate_router_case(case: dict) -> dict:
    actual_tools = select_tool_names(
        case["question"],
        case.get("context")
    )
    expected_tools = case["expected_tools"]

    return {
        "id": case["id"],
        "passed": actual_tools == expected_tools,
        "expected": expected_tools,
        "actual": actual_tools
    }


def evaluate_keyword_rag_case(case: dict) -> dict:
    result = search_knowledge_base(case["question"])
    actual_chunk_ids = [
        item["chunk_id"]
        for item in result["results"]
    ]
    expected_chunk_ids = case["expected_chunk_ids"]

    return {
        "id": case["id"],
        "passed": all(
            chunk_id in actual_chunk_ids
            for chunk_id in expected_chunk_ids
        ),
        "expected": expected_chunk_ids,
        "actual": actual_chunk_ids
    }


def evaluate_data_tool_case(case: dict) -> dict:
    tool = TOOL_REGISTRY[case["tool_name"]]
    result = tool(**case["arguments"])
    expected_fields = case["expected_fields"]
    actual_fields = {
        key: result.get(key)
        for key in expected_fields
    }

    return {
        "id": case["id"],
        "passed": actual_fields == expected_fields,
        "expected": expected_fields,
        "actual": actual_fields
    }


def build_sample_memory_messages() -> list:
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": "第一轮回答"},
        {"role": "user", "content": "第二轮问题"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "query_teacher_schedule",
                        "arguments": {"teacher_name": "李老师", "date": "周五"}
                    }
                }
            ]
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        {"role": "assistant", "content": "第二轮回答"},
        {"role": "user", "content": "第三轮问题"},
        {"role": "assistant", "content": "第三轮回答"}
    ]


def evaluate_memory_case(case: dict) -> dict:
    messages = build_sample_memory_messages()
    trim_messages(messages, max_messages=case["max_messages"])

    actual_roles = [
        message["role"]
        for message in messages
    ]
    actual_user_messages = [
        message["content"]
        for message in messages
        if message["role"] == "user"
    ]

    actual = {
        "roles": actual_roles,
        "user_messages": actual_user_messages
    }
    expected = {
        "roles": case["expected_roles"],
        "user_messages": case["expected_user_messages"]
    }

    return {
        "id": case["id"],
        "passed": actual == expected,
        "expected": expected,
        "actual": actual
    }


def evaluate_streaming_case(case: dict) -> dict:
    messages = agent_core.create_initial_messages()
    original_stream_llm = agent_core.stream_llm

    def fake_stream_llm(messages: list, tools: list):
        for chunk in case["expected_chunks"]:
            yield {
                "type": "content_delta",
                "delta": chunk
            }

        yield {
            "type": "assistant_message",
            "message": {
                "role": "assistant",
                "content": case["expected_answer"]
            }
        }

    try:
        agent_core.stream_llm = fake_stream_llm
        chunks = list(
            agent_core.run_agent_stream(
                messages,
                case["question"],
                context={
                    "user_role": "student",
                    "page": "general"
                }
            )
        )
    finally:
        agent_core.stream_llm = original_stream_llm

    actual = {
        "chunks": chunks,
        "answer": "".join(chunks),
        "last_message": messages[-1]
    }
    expected = {
        "chunks": case["expected_chunks"],
        "answer": case["expected_answer"],
        "last_message": {
            "role": "assistant",
            "content": case["expected_answer"]
        }
    }

    return {
        "id": case["id"],
        "passed": actual == expected,
        "expected": expected,
        "actual": actual
    }


def evaluate_streaming_tool_call_case(case: dict) -> dict:
    tool_calls_by_index = {}
    append_tool_call_delta(
        tool_calls_by_index,
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "query_teacher_schedule",
                "arguments": "{\"teacher_name\":\"李"
            }
        }
    )
    append_tool_call_delta(
        tool_calls_by_index,
        {
            "index": 0,
            "function": {
                "arguments": "老师\",\"date\":\"周五\"}"
            }
        }
    )
    assistant_message = build_streamed_assistant_message(
        "assistant",
        [],
        tool_calls_by_index
    )

    actual = assistant_message["tool_calls"]
    expected = case["expected_tool_calls"]

    return {
        "id": case["id"],
        "passed": actual == expected,
        "expected": expected,
        "actual": actual
    }


def evaluate_case(case: dict) -> dict:
    if case["type"] == "router":
        return evaluate_router_case(case)

    if case["type"] == "keyword_rag":
        return evaluate_keyword_rag_case(case)

    if case["type"] == "data_tool":
        return evaluate_data_tool_case(case)

    if case["type"] == "memory":
        return evaluate_memory_case(case)

    if case["type"] == "streaming":
        return evaluate_streaming_case(case)

    if case["type"] == "streaming_tool_call":
        return evaluate_streaming_tool_call_case(case)

    return {
        "id": case["id"],
        "passed": False,
        "expected": case["type"],
        "actual": "unknown eval case type"
    }


def main() -> None:
    results = [
        evaluate_case(case)
        for case in load_eval_cases()
    ]
    passed_count = sum(1 for result in results if result["passed"])

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['id']}")
        print(f"  expected: {result['expected']}")
        print(f"  actual:   {result['actual']}")

    print()
    print(f"Passed {passed_count}/{len(results)} eval cases.")

    if passed_count != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
