import asyncio
import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from config.settings import settings
from models.chat import ChatSession, ChatMessage, MessageRole

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    user_id     TEXT,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chat_messages (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    id          TEXT NOT NULL UNIQUE,
    session_id  TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON chat_sessions(user_id, updated_at);
"""


class SQLiteDatabase:
    """
    Local SQLite chat store (replaces Supabase / MongoDB).

    Exposes the same async interface as the previous database classes so the
    rest of the app does not need to change. Sessions and messages live in
    two tables in a single file (``DATABASE_PATH``, default ``data/chatbot.db``).
    Blocking sqlite3 calls run in a worker thread to keep the event loop free.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or settings.database_path
        self.conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    async def connect(self):
        """Open the database file and create tables if needed."""
        try:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.executescript(SCHEMA)
            self.conn.commit()
            logger.info(f"Connected to SQLite database at {self.path}")
        except Exception as e:
            logger.error(f"Failed to open SQLite database: {e}")
            raise

    async def disconnect(self):
        """Close the database file."""
        if self.conn:
            self.conn.close()
            self.conn = None
        logger.info("Disconnected from SQLite database")

    async def _run(self, fn, *args):
        """Run a blocking DB function in a thread, one at a time."""
        async with self._lock:
            return await asyncio.to_thread(fn, *args)

    def _load_session(self, row: sqlite3.Row) -> ChatSession:
        messages = [
            ChatMessage(
                id=m["id"],
                role=m["role"],
                content=m["content"],
                timestamp=m["timestamp"],
                metadata=json.loads(m["metadata"]),
            )
            for m in self.conn.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY seq",
                (row["session_id"],),
            )
        ]
        return ChatSession(
            session_id=row["session_id"],
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            title=row["title"],
            messages=messages,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"]),
        )

    async def create_chat_session(
        self,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        tenant_id: str = "default",
        collection_prefix: str = "",
    ) -> str:
        """Create a new chat session and return its id."""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        title = title or f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"

        def insert():
            self.conn.execute(
                "INSERT INTO chat_sessions (session_id, tenant_id, user_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, tenant_id, user_id, title, now, now),
            )
            self.conn.commit()

        await self._run(insert)
        logger.info(f"Created new chat session: {session_id}")
        return session_id

    async def add_message_to_session(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        collection_prefix: str = "",
    ) -> bool:
        """Append a message to an existing chat session."""
        message = ChatMessage(role=role, content=content, metadata=metadata or {})

        def insert() -> bool:
            exists = self.conn.execute(
                "SELECT 1 FROM chat_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if not exists:
                return False
            self.conn.execute(
                "INSERT INTO chat_messages (id, session_id, role, content, timestamp, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    message.id,
                    session_id,
                    message.role.value,
                    message.content,
                    message.timestamp.isoformat(),
                    json.dumps(message.model_dump(mode="json")["metadata"]),
                ),
            )
            self.conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                (datetime.utcnow().isoformat(), session_id),
            )
            self.conn.commit()
            return True

        try:
            added = await self._run(insert)
            if not added:
                logger.warning(f"Session {session_id} not found")
            return added
        except Exception as e:
            logger.error(f"Failed to add message to session {session_id}: {e}")
            return False

    async def get_chat_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by id."""
        def fetch():
            row = self.conn.execute(
                "SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            return self._load_session(row) if row else None

        try:
            return await self._run(fetch)
        except Exception as e:
            logger.error(f"Failed to get chat session {session_id}: {e}")
            return None

    async def get_user_chat_sessions(self, user_id: str, limit: int = 50) -> List[ChatSession]:
        """Get all chat sessions for a user, newest first."""
        def fetch():
            rows = self.conn.execute(
                "SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
            return [self._load_session(r) for r in rows]

        try:
            return await self._run(fetch)
        except Exception as e:
            logger.error(f"Failed to get user chat sessions: {e}")
            return []

    async def get_all_chat_sessions(self, limit: int = 50) -> List[ChatSession]:
        """Get all chat sessions (admin), newest first."""
        def fetch():
            rows = self.conn.execute(
                "SELECT * FROM chat_sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._load_session(r) for r in rows]

        try:
            return await self._run(fetch)
        except Exception as e:
            logger.error(f"Failed to get all chat sessions: {e}")
            return []

    async def count_chat_sessions(self) -> int:
        """Total number of stored chat sessions."""
        def count():
            return self.conn.execute("SELECT COUNT(*) FROM chat_sessions").fetchone()[0]

        return await self._run(count)

    async def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session and its messages."""
        def delete() -> bool:
            cur = self.conn.execute(
                "DELETE FROM chat_sessions WHERE session_id = ?", (session_id,)
            )
            self.conn.commit()
            return cur.rowcount > 0

        try:
            deleted = await self._run(delete)
            if deleted:
                logger.info(f"Deleted chat session: {session_id}")
            return deleted
        except Exception as e:
            logger.error(f"Failed to delete chat session {session_id}: {e}")
            return False


# Global database instance
db = SQLiteDatabase()
