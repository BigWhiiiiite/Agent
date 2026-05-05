from contextlib import contextmanager
from threading import Lock, RLock
from typing import Iterator

from agent.core import create_initial_messages


SESSIONS: dict[str, list] = {}
SESSION_LOCKS: dict[str, RLock] = {}
STORE_LOCK = Lock()


def get_session_lock(session_id: str) -> RLock:
    with STORE_LOCK:
        if session_id not in SESSION_LOCKS:
            SESSION_LOCKS[session_id] = RLock()

        return SESSION_LOCKS[session_id]


@contextmanager
def use_session_messages(
    session_id: str,
    reset: bool = False
) -> Iterator[list]:
    session_lock = get_session_lock(session_id)

    with session_lock:
        with STORE_LOCK:
            if reset or session_id not in SESSIONS:
                SESSIONS[session_id] = create_initial_messages()

        yield SESSIONS[session_id]


def get_session_messages(session_id: str) -> list:
    with use_session_messages(session_id) as messages:
        return messages


def reset_session(session_id: str) -> list:
    with use_session_messages(session_id, reset=True) as messages:
        return messages


def delete_session(session_id: str) -> bool:
    session_lock = get_session_lock(session_id)

    with session_lock:
        with STORE_LOCK:
            if session_id not in SESSIONS:
                return False

            del SESSIONS[session_id]
            return True


def list_session_ids() -> list[str]:
    with STORE_LOCK:
        return list(SESSIONS.keys())
