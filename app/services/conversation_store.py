import json
import sqlite3

from app.services.db import get_connection

WINDOW_SIZE = 10


def _get_connection() -> sqlite3.Connection:
    # O schema (tabela `messages` + índice) é criado por get_connection(), em db.py.
    return get_connection()


def get_recent_messages(phone_number: str, with_tools: bool = False) -> list[dict]:
    """Janela recente da conversa. `with_tools=True` inclui a chave `tools`
    nas respostas do assistente: lista de `{"nome","args","ok"}`, `[]` quando
    a resposta não chamou ferramenta e `None` em mensagens anteriores ao
    registro de ferramentas."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT role, content, tools FROM messages WHERE phone_number = ? "
            "ORDER BY id DESC LIMIT ?",
            (phone_number, WINDOW_SIZE),
        ).fetchall()
    finally:
        conn.close()
    mensagens = []
    for role, content, tools in reversed(rows):
        msg = {"role": role, "content": content}
        if with_tools and role == "assistant":
            msg["tools"] = json.loads(tools) if tools is not None else None
        mensagens.append(msg)
    return mensagens


def add_message(
    phone_number: str, role: str, content: str, tools: list[dict] | None = None
) -> None:
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO messages (phone_number, role, content, tools) VALUES (?, ?, ?, ?)",
            (
                phone_number, role, content,
                json.dumps(tools, ensure_ascii=False) if tools is not None else None,
            ),
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
