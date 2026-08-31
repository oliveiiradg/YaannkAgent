import sqlite3
from pathlib import Path

DB_PATH = Path.home() / "YaannkAgent" / "gateway" / "data" / "conversations.db"

# Schema único do banco. Criado on-demand a cada get_connection() — todos os
# CREATE são `IF NOT EXISTS`, então é barato e idempotente.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phone_number TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_messages_phone ON messages(phone_number, id);

CREATE TABLE IF NOT EXISTS expenses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT,
    autor TEXT,
    valor REAL,
    categoria TEXT,
    descricao TEXT,
    data TEXT,          -- ISO8601 (YYYY-MM-DD): data do gasto, nem sempre hoje
    created_at TEXT     -- ISO8601: momento do registro
);
CREATE INDEX IF NOT EXISTS idx_expenses_chat_data ON expenses(chat_id, data);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)
    return conn
