import json
import sys
import threading
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import agent.core as agent_core
import agent.data_store as data_store
import agent.router as router_module
from agent.llm import append_tool_call_delta, build_streamed_assistant_message
from agent.memory import SUMMARY_PREFIX, trim_messages
from agent.rag import search_knowledge_base
from agent.router import select_tool_names
from agent.session_store import use_session_messages
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


def evaluate_tool_selector_case(case: dict) -> dict:
    original_selector = router_module.choose_tool_names_with_llm
    original_enabled = router_module.USE_LLM_TOOL_SELECTOR
    original_min_tools = router_module.TOOL_SELECTOR_MIN_TOOLS

    def fake_tool_selector(
        user_input: str,
        context: dict,
        tool_catalog: list[dict]
    ) -> list:
        return case["llm_selected_tools"]

    try:
        router_module.choose_tool_names_with_llm = fake_tool_selector
        router_module.USE_LLM_TOOL_SELECTOR = True
        router_module.TOOL_SELECTOR_MIN_TOOLS = 1
        actual_tool_names = router_module.select_tool_names_with_llm(
            case["question"],
            case.get("context")
        )
    finally:
        router_module.choose_tool_names_with_llm = original_selector
        router_module.USE_LLM_TOOL_SELECTOR = original_enabled
        router_module.TOOL_SELECTOR_MIN_TOOLS = original_min_tools

    expected_tool_names = case["expected_tools"]

    return {
        "id": case["id"],
        "passed": actual_tool_names == expected_tool_names,
        "expected": expected_tool_names,
        "actual": actual_tool_names
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


def evaluate_data_provider_case(case: dict) -> dict:
    original_provider = data_store.get_data_provider()

    class FakeDataProvider:
        def get_teacher_schedule(self, teacher_name: str, date: str) -> dict:
            return {
                "teacher_name": teacher_name,
                "date": date,
                "available_slots": case["expected_schedule_slots"]
            }

        def get_course_info(self, course_name: str) -> dict:
            return {
                "teacher": case["expected_course_teacher"],
                "time": "周一 09:00",
                "classroom": "T101",
                "description": "测试课程"
            }

    try:
        data_store.set_data_provider(FakeDataProvider())
        schedule_result = TOOL_REGISTRY["query_teacher_schedule"]("任意老师", "任意日期")
        course_result = TOOL_REGISTRY["query_course_info"]("任意课程")
    finally:
        data_store.set_data_provider(original_provider)

    actual = {
        "schedule_slots": schedule_result["available_slots"],
        "course_teacher": course_result["teacher"]
    }
    expected = {
        "schedule_slots": case["expected_schedule_slots"],
        "course_teacher": case["expected_course_teacher"]
    }

    return {
        "id": case["id"],
        "passed": actual == expected,
        "expected": expected,
        "actual": actual
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


def build_long_memory_messages() -> list:
    long_text = "很长的上下文" * 80

    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "第一轮长问题：" + long_text},
        {"role": "assistant", "content": "第一轮回答"},
        {"role": "user", "content": "第二轮长问题：" + long_text},
        {"role": "assistant", "content": "第二轮回答"},
        {"role": "user", "content": "第三轮短问题"},
        {"role": "assistant", "content": "第三轮回答"}
    ]


def evaluate_memory_case(case: dict) -> dict:
    if case.get("fixture") == "long_memory":
        messages = build_long_memory_messages()
    else:
        messages = build_sample_memory_messages()

    def fake_summary_builder(
        previous_summary: str,
        removed_turns: list[list],
        max_chars: int
    ) -> str:
        return (
            "LLM摘要\n"
            "用户目标/问题：第一轮问题已经讨论过。\n"
            "已知事实：后续问题可能会引用这段历史。\n"
            "用户偏好：无。\n"
            "工具结果：无。\n"
            "未完成事项：无。"
        )

    trim_messages(
        messages,
        max_messages=case["max_messages"],
        max_context_chars=case.get("max_context_chars", 1000000),
        max_recent_turns=case["max_recent_turns"],
        summary_builder=fake_summary_builder
    )

    actual_roles = [
        message["role"]
        for message in messages
    ]
    actual_user_messages = [
        message["content"]
        for message in messages
        if message["role"] == "user"
    ]
    summary_messages = [
        message["content"]
        for message in messages
        if (
            message["role"] == "system"
            and message.get("content", "").startswith(SUMMARY_PREFIX)
        )
    ]
    actual_summary_contains = [
        expected_text
        for expected_text in case["expected_summary_contains"]
        if any(expected_text in summary for summary in summary_messages)
    ]

    actual = {
        "roles": actual_roles,
        "user_messages": actual_user_messages,
        "summary_contains": actual_summary_contains
    }
    expected = {
        "roles": case["expected_roles"],
        "user_messages": case["expected_user_messages"],
        "summary_contains": case["expected_summary_contains"]
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
    original_select_tools = agent_core.select_tools

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
        agent_core.select_tools = lambda user_input, context=None: []
        trace_holder = {}
        chunks = list(
            agent_core.run_agent_stream(
                messages,
                case["question"],
                context={
                    "user_role": "student",
                    "page": "general"
                },
                trace_callback=lambda trace: trace_holder.update({"trace": trace})
            )
        )
    finally:
        agent_core.stream_llm = original_stream_llm
        agent_core.select_tools = original_select_tools

    actual = {
        "chunks": chunks,
        "answer": "".join(chunks),
        "last_message": messages[-1],
        "trace_answer": trace_holder["trace"]["final_answer"],
        "trace_stop_reason": trace_holder["trace"]["stop_reason"]
    }
    expected = {
        "chunks": case["expected_chunks"],
        "answer": case["expected_answer"],
        "last_message": {
            "role": "assistant",
            "content": case["expected_answer"]
        },
        "trace_answer": case["expected_answer"],
        "trace_stop_reason": "final_answer"
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


def evaluate_session_lock_case(case: dict) -> dict:
    events = []
    first_started = threading.Event()

    def first_worker() -> None:
        with use_session_messages(case["session_id"], reset=True):
            events.append("first-start")
            first_started.set()
            time.sleep(0.05)
            events.append("first-end")

    def second_worker() -> None:
        first_started.wait(timeout=1)
        with use_session_messages(case["session_id"]):
            events.append("second-start")
            events.append("second-end")

    first_thread = threading.Thread(target=first_worker)
    second_thread = threading.Thread(target=second_worker)

    first_thread.start()
    second_thread.start()
    first_thread.join()
    second_thread.join()

    return {
        "id": case["id"],
        "passed": events == case["expected_events"],
        "expected": case["expected_events"],
        "actual": events
    }


def evaluate_case(case: dict) -> dict:
    if case["type"] == "router":
        return evaluate_router_case(case)

    if case["type"] == "tool_selector":
        return evaluate_tool_selector_case(case)

    if case["type"] == "keyword_rag":
        return evaluate_keyword_rag_case(case)

    if case["type"] == "data_tool":
        return evaluate_data_tool_case(case)

    if case["type"] == "data_provider":
        return evaluate_data_provider_case(case)

    if case["type"] == "memory":
        return evaluate_memory_case(case)

    if case["type"] == "streaming":
        return evaluate_streaming_case(case)

    if case["type"] == "streaming_tool_call":
        return evaluate_streaming_tool_call_case(case)

    if case["type"] == "session_lock":
        return evaluate_session_lock_case(case)

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
