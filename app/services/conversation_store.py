import sqlite3

from app.services.db import get_connection

WINDOW_SIZE = 10


def _get_connection() -> sqlite3.Connection:
    # O schema (tabela `messages` + índice) é criado por get_connection(), em db.py.
    return get_connection()


def get_recent_messages(phone_number: str) -> list[dict[str, str]]:
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE phone_number = ? "
            "ORDER BY id DESC LIMIT ?",
            (phone_number, WINDOW_SIZE),
        ).fetchall()
    finally:
        conn.close()
    return [{"role": role, "content": content} for role, content in reversed(rows)]


def add_message(phone_number: str, role: str, content: str) -> None:
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO messages (phone_number, role, content) VALUES (?, ?, ?)",
            (phone_number, role, content),
        )
        conn.execute(
            """
            DELETE FROM messages
            WHERE phone_number = ? AND id NOT IN (
                SELECT id FROM messages WHERE phone_number = ?
                ORDER BY id DESC LIMIT ?
            )
            """,
            (phone_number, phone_number, WINDOW_SIZE),
        )
        conn.commit()
    finally:
        conn.close()


def clear_history(phone_number: str) -> None:
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM messages WHERE phone_number = ?", (phone_number,))
        conn.commit()
    finally:
        conn.close()
