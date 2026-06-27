import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from supabase import create_async_client, AsyncClient

from config.settings import settings
from models.chat import ChatSession, ChatMessage, MessageRole

logger = logging.getLogger(__name__)

TABLE = "chat_sessions"


class SupabaseDatabase:
    """
    Supabase-backed chat store (replaces MongoDatabase).

    Drop-in replacement for the previous MongoDatabase: it exposes the exact
    same async interface so the rest of the app does not need to change.

    Talks to Supabase through the Data API (PostgREST) using the project's
    secret key, which bypasses RLS — so no database password is required.

    A whole chat session is stored as a single row in ``public.chat_sessions``
    with the message list kept in a ``jsonb`` column, mirroring the original
    Mongo document model so ``ChatSession(**row)`` keeps working.
    """

    def __init__(self):
        self.client: Optional[AsyncClient] = None

    async def connect(self):
        """Create the Supabase client and verify connectivity."""
        try:
            if not settings.supabase_url or not settings.supabase_secret_key:
                raise RuntimeError(
                    "SUPABASE_URL / SUPABASE_SECRET_KEY are not set. Add them to .env."
                )

            self.client = await create_async_client(
                settings.supabase_url,
                settings.supabase_secret_key,
            )

            # Verify the table is reachable with the provided key.
            await self.client.table(TABLE).select("session_id").limit(1).execute()
            logger.info("Successfully connected to Supabase")

        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            raise

    async def disconnect(self):
        """No persistent connection to close for the Data API client."""
        logger.info("Disconnected from Supabase")

    async def create_chat_session(
        self,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        tenant_id: str = "default",
        collection_prefix: str = "",
    ) -> str:
        """Create a new chat session and return its id."""
        session_id = str(uuid.uuid4())
        session = ChatSession(
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            title=title or f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        )

        # mode="json" makes datetimes/enums JSON-serializable for the Data API.
        await self.client.table(TABLE).insert(session.model_dump(mode="json")).execute()

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
        try:
            message = ChatMessage(
                role=role,
                content=content,
                metadata=metadata or {},
            )

            # Read-modify-write the jsonb messages array (PostgREST has no
            # atomic array append). Concurrency per session is sequential here.
            res = (
                await self.client.table(TABLE)
                .select("messages")
                .eq("session_id", session_id)
                .execute()
            )
            if not res.data:
                logger.warning(f"Session {session_id} not found")
                return False

            messages = res.data[0].get("messages") or []
            messages.append(message.model_dump(mode="json"))

            await (
                self.client.table(TABLE)
                .update(
                    {
                        "messages": messages,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                )
                .eq("session_id", session_id)
                .execute()
            )

            logger.info(f"Added message to session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add message to session {session_id}: {e}")
            return False

    async def get_chat_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by id."""
        try:
            res = (
                await self.client.table(TABLE)
                .select("*")
                .eq("session_id", session_id)
                .limit(1)
                .execute()
            )
            if res.data:
                return ChatSession(**res.data[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get chat session {session_id}: {e}")
            return None

    async def get_user_chat_sessions(self, user_id: str, limit: int = 50) -> List[ChatSession]:
        """Get all chat sessions for a user, newest first."""
        try:
            res = (
                await self.client.table(TABLE)
                .select("*")
                .eq("user_id", user_id)
                .order("updated_at", desc=True)
                .limit(limit)
                .execute()
            )
            return [ChatSession(**row) for row in res.data]
        except Exception as e:
            logger.error(f"Failed to get user chat sessions: {e}")
            return []

    async def get_all_chat_sessions(self, limit: int = 50) -> List[ChatSession]:
        """Get all chat sessions (admin), newest first."""
        try:
            res = (
                await self.client.table(TABLE)
                .select("*")
                .order("updated_at", desc=True)
                .limit(limit)
                .execute()
            )
            return [ChatSession(**row) for row in res.data]
        except Exception as e:
            logger.error(f"Failed to get all chat sessions: {e}")
            return []

    async def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        try:
            res = (
                await self.client.table(TABLE)
                .delete()
                .eq("session_id", session_id)
                .execute()
            )
            if res.data:
                logger.info(f"Deleted chat session: {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete chat session {session_id}: {e}")
            return False


# Global database instance
db = SupabaseDatabase()
