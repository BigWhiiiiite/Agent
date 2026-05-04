import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from agent.rag import search_knowledge_base
from agent.router import select_tool_names


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


def evaluate_case(case: dict) -> dict:
    if case["type"] == "router":
        return evaluate_router_case(case)

    if case["type"] == "keyword_rag":
        return evaluate_keyword_rag_case(case)

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
