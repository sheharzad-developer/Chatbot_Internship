import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import logging

from config.settings import settings
from models.chat import ChatSession, ChatMessage, MessageRole

logger = logging.getLogger(__name__)

class MongoDatabase:
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None
        self.chat_sessions_collection = None
        
    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(settings.mongodb_url)
            self.database = self.client[settings.mongodb_db_name]
            self.chat_sessions_collection = self.database.chat_sessions
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def create_chat_session(self, user_id: Optional[str] = None, title: Optional[str] = None, tenant_id: str = "default", collection_prefix: str = "") -> str:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        session = ChatSession(
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            title=title or f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        )
        
        await self.chat_sessions_collection.insert_one(session.dict())
        logger.info(f"Created new chat session: {session_id}")
        return session_id
    
    async def add_message_to_session(self, session_id: str, role: MessageRole, content: str, metadata: Optional[Dict[str, Any]] = None, collection_prefix: str = "") -> bool:
        """Add a message to an existing chat session"""
        try:
            message = ChatMessage(
                role=role,
                content=content,
                metadata=metadata or {}
            )
            
            result = await self.chat_sessions_collection.update_one(
                {"session_id": session_id},
                {
                    "$push": {"messages": message.dict()},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Added message to session {session_id}")
                return True
            else:
                logger.warning(f"Session {session_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to add message to session {session_id}: {e}")
            return False
    
    async def get_chat_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID"""
        try:
            doc = await self.chat_sessions_collection.find_one({"session_id": session_id})
            if doc:
                return ChatSession(**doc)
            return None
        except Exception as e:
            logger.error(f"Failed to get chat session {session_id}: {e}")
            return None
    
    async def get_user_chat_sessions(self, user_id: str, limit: int = 50) -> List[ChatSession]:
        """Get all chat sessions for a user"""
        try:
            cursor = self.chat_sessions_collection.find(
                {"user_id": user_id}
            ).sort("updated_at", -1).limit(limit)
            
            sessions = []
            async for doc in cursor:
                sessions.append(ChatSession(**doc))
            
            return sessions
        except Exception as e:
            logger.error(f"Failed to get user chat sessions: {e}")
            return []
    
    async def get_all_chat_sessions(self, limit: int = 50) -> List[ChatSession]:
        """Get all chat sessions (for admin purposes)"""
        try:
            cursor = self.chat_sessions_collection.find().sort("updated_at", -1).limit(limit)
            
            sessions = []
            async for doc in cursor:
                sessions.append(ChatSession(**doc))
            
            return sessions
        except Exception as e:
            logger.error(f"Failed to get all chat sessions: {e}")
            return []
    
    async def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session"""
        try:
            result = await self.chat_sessions_collection.delete_one({"session_id": session_id})
            if result.deleted_count > 0:
                logger.info(f"Deleted chat session: {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete chat session {session_id}: {e}")
            return False

# Global database instance
db = MongoDatabase() 