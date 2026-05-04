from agent.core import create_initial_messages


SESSIONS: dict[str, list] = {}


def get_session_messages(session_id: str) -> list:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = create_initial_messages()

    return SESSIONS[session_id]


def reset_session(session_id: str) -> list:
    SESSIONS[session_id] = create_initial_messages()
    return SESSIONS[session_id]


def delete_session(session_id: str) -> bool:
    if session_id not in SESSIONS:
        return False

    del SESSIONS[session_id]
    return True


def list_session_ids() -> list[str]:
    return list(SESSIONS.keys())
