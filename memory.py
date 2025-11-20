"""PostgreSQL integration for chat history and session memory."""
from typing import List, Dict

# Try to import PostgreSQL dependencies, but make them optional
try:
    from langchain_postgres import PostgresChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    # Silently skip - PostgreSQL is optional

from config import POSTGRES_CONNECTION_STRING

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


def initialize_database():
    """
    Initialize PostgreSQL database and create necessary tables.
    This should be run once during setup.
    """
    if not PSYCOPG2_AVAILABLE:
        print("⚠ PostgreSQL not available. Skipping database initialization.")
        return
    
    print("Initializing PostgreSQL database...")
    
    try:
        # Import config values
        from config import POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
        
        # Connect to default postgres database
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database='postgres'  # Connect to default database first
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{POSTGRES_DB}'")
        if not cursor.fetchone():
            cursor.execute(f"CREATE DATABASE {POSTGRES_DB}")
            print(f"✓ Created database: {POSTGRES_DB}")
        else:
            print(f"✓ Database already exists: {POSTGRES_DB}")
        
        cursor.close()
        conn.close()
        print("✓ Database initialization complete")
    except Exception as e:
        print(f"⚠ Database initialization warning: {str(e)}")
        print("  You may need to create the database manually or update PostgreSQL settings in .env")


def load_chat_history(session_id: str) -> List[Dict[str, str]]:
    """
    Load chat history for a session from PostgreSQL.
    
    Args:
        session_id: Session identifier
        
    Returns:
        List of chat messages
    """
    if not POSTGRES_AVAILABLE:
        return []
    
    try:
        history = PostgresChatMessageHistory(
            connection_string=POSTGRES_CONNECTION_STRING,
            session_id=session_id
        )
        
        messages = []
        for msg in history.messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        
        return messages
    except Exception as e:
        print(f"⚠ Could not load chat history: {str(e)}")
        return []


def save_to_chat_history(session_id: str, user_message: str, assistant_message: str):
    """
    Save a conversation turn to PostgreSQL.
    
    Args:
        session_id: Session identifier
        user_message: User's message
        assistant_message: Assistant's response
    """
    if not POSTGRES_AVAILABLE:
        return
    
    try:
        history = PostgresChatMessageHistory(
            connection_string=POSTGRES_CONNECTION_STRING,
            session_id=session_id
        )
        
        history.add_user_message(user_message)
        history.add_ai_message(assistant_message)
        
        print(f"✓ Saved conversation to session: {session_id}")
    except Exception as e:
        print(f"⚠ Could not save to chat history: {str(e)}")


def clear_chat_history(session_id: str):
    """
    Clear chat history for a session.
    
    Args:
        session_id: Session identifier
    """
    if not POSTGRES_AVAILABLE:
        return
    
    try:
        history = PostgresChatMessageHistory(
            connection_string=POSTGRES_CONNECTION_STRING,
            session_id=session_id
        )
        history.clear()
        print(f"✓ Cleared chat history for session: {session_id}")
    except Exception as e:
        print(f"⚠ Could not clear chat history: {str(e)}")


if __name__ == "__main__":
    # Initialize database when run directly
    initialize_database()
