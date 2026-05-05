from collections.abc import Callable

from agent.config import (
    MAX_CONTEXT_MESSAGES,
    MAX_RECENT_TURNS,
    MAX_SUMMARY_CHARS,
)


SUMMARY_PREFIX = "对话摘要："
SummaryBuilder = Callable[[str, list[list], int], str]


def is_summary_message(message: dict) -> bool:
    content = message.get("content") or ""
    return (
        message.get("role") == "system"
        and content.startswith(SUMMARY_PREFIX)
    )


def split_memory_sections(messages: list) -> tuple[list, dict | None, list[list]]:
    system_messages = []
    summary_message = None
    index = 0

    while index < len(messages) and messages[index].get("role") == "system":
        message = messages[index]

        if is_summary_message(message):
            summary_message = message
        else:
            system_messages.append(message)

        index += 1

    turns = []
    current_turn = []

    for message in messages[index:]:
        if message.get("role") == "user":
            if current_turn:
                turns.append(current_turn)
            current_turn = [message]
            continue

        if current_turn:
            current_turn.append(message)

    if current_turn:
        turns.append(current_turn)

    return system_messages, summary_message, turns


def split_system_and_turns(messages: list) -> tuple[list, list[list]]:
    system_messages, summary_message, turns = split_memory_sections(messages)

    if summary_message:
        system_messages = system_messages + [summary_message]

    return system_messages, turns


def clip_text(text: str, max_chars: int = 80) -> str:
    compact_text = " ".join(str(text).split())

    if len(compact_text) <= max_chars:
        return compact_text

    return compact_text[:max_chars] + "..."


def collect_user_questions(turns: list[list]) -> list[str]:
    questions = []

    for turn in turns:
        for message in turn:
            if message.get("role") == "user":
                questions.append(clip_text(message.get("content", "")))

    return questions


def collect_tool_names(turns: list[list]) -> list[str]:
    tool_names = []

    for turn in turns:
        for message in turn:
            for tool_call in message.get("tool_calls", []):
                function_info = tool_call.get("function", {})
                tool_name = function_info.get("name")

                if tool_name and tool_name not in tool_names:
                    tool_names.append(tool_name)

    return tool_names


def build_rule_summary(turns: list[list]) -> str:
    questions = collect_user_questions(turns)
    tool_names = collect_tool_names(turns)
    summary_parts = []

    if questions:
        summary_parts.append("用户之前问过：" + "；".join(questions[-6:]))

    if tool_names:
        summary_parts.append("历史中调用过工具：" + "、".join(tool_names[-8:]))

    return "。".join(summary_parts)


def normalize_summary_text(summary_text: str) -> str:
    summary_text = str(summary_text).strip()
    summary_text = summary_text.removeprefix(SUMMARY_PREFIX).strip()
    return summary_text


def clip_summary_text(summary_text: str) -> str:
    if len(summary_text) <= MAX_SUMMARY_CHARS:
        return summary_text

    return summary_text[-MAX_SUMMARY_CHARS:]


def build_fallback_summary_text(
    previous_summary: str,
    removed_turns: list[list]
) -> str:
    new_summary = build_rule_summary(removed_turns)
    summary_parts = [
        part
        for part in [previous_summary, new_summary]
        if part
    ]

    return "。".join(summary_parts)


def build_summary_text(
    previous_summary: str,
    removed_turns: list[list],
    summary_builder: SummaryBuilder | None = None
) -> str:
    if summary_builder:
        try:
            llm_summary = summary_builder(
                previous_summary,
                removed_turns,
                MAX_SUMMARY_CHARS
            )
            llm_summary = normalize_summary_text(llm_summary)

            if llm_summary:
                return llm_summary
        except Exception:
            pass

    return build_fallback_summary_text(previous_summary, removed_turns)


def build_summary_message(
    previous_summary_message: dict | None,
    removed_turns: list[list],
    summary_builder: SummaryBuilder | None = None
) -> dict | None:
    previous_summary = ""

    if previous_summary_message:
        previous_summary = previous_summary_message.get("content", "")
        previous_summary = normalize_summary_text(previous_summary)

    summary_text = build_summary_text(
        previous_summary,
        removed_turns,
        summary_builder
    )

    if not summary_text:
        return None

    return {
        "role": "system",
        "content": SUMMARY_PREFIX + " " + clip_summary_text(summary_text)
    }


def flatten_turns(turns: list[list]) -> list:
    return [
        message
        for turn in turns
        for message in turn
    ]


def trim_messages(
    messages: list,
    max_messages: int = MAX_CONTEXT_MESSAGES,
    max_recent_turns: int = MAX_RECENT_TURNS,
    summary_builder: SummaryBuilder | None = None
) -> int:
    if len(messages) <= max_messages:
        return 0

    system_messages, summary_message, turns = split_memory_sections(messages)

    if not turns:
        return 0

    previous_summary_message = summary_message
    removed_turns = turns[:-max_recent_turns]
    kept_turns = turns[-max_recent_turns:]

    def estimate_new_message_count() -> int:
        summary_count = 1 if previous_summary_message or removed_turns else 0
        return (
            len(system_messages)
            + summary_count
            + len(flatten_turns(kept_turns))
        )

    while len(kept_turns) > 1 and estimate_new_message_count() > max_messages:
        removed_turns.append(kept_turns.pop(0))

    summary_message = build_summary_message(
        previous_summary_message,
        removed_turns,
        summary_builder
    )

    removed_message_count = len(flatten_turns(removed_turns))
    summary_messages = [summary_message] if summary_message else []
    messages[:] = system_messages + summary_messages + flatten_turns(kept_turns)

    return removed_message_count
