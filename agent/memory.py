from agent.config import MAX_CONTEXT_MESSAGES


def split_system_and_turns(messages: list) -> tuple[list, list[list]]:
    system_messages = []
    index = 0

    while index < len(messages) and messages[index].get("role") == "system":
        system_messages.append(messages[index])
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

    return system_messages, turns


def flatten_turns(turns: list[list]) -> list:
    return [
        message
        for turn in turns
        for message in turn
    ]


def trim_messages(
    messages: list,
    max_messages: int = MAX_CONTEXT_MESSAGES
) -> int:
    if len(messages) <= max_messages:
        return 0

    system_messages, turns = split_system_and_turns(messages)

    if not turns:
        return 0

    history_budget = max_messages - len(system_messages)

    if history_budget <= 0:
        original_count = len(messages)
        messages[:] = system_messages[:max_messages]
        return original_count - len(messages)

    kept_turns = []
    kept_count = 0

    for turn in reversed(turns):
        turn_length = len(turn)

        if kept_turns and kept_count + turn_length > history_budget:
            break

        kept_turns.append(turn)
        kept_count += turn_length

    kept_turns.reverse()

    original_count = len(messages)
    messages[:] = system_messages + flatten_turns(kept_turns)

    return original_count - len(messages)
