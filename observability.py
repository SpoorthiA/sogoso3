"""Langfuse integration for observability and tracing."""
import os
import warnings
import logging

# Suppress ALL Langfuse-related output
warnings.filterwarnings("ignore", module="langfuse")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["LANGFUSE_DEBUG"] = "False"

# Suppress Langfuse SDK logging completely
logging.getLogger("langfuse").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)

from langfuse import Langfuse
from config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
from contextlib import contextmanager
from typing import Any, Dict, Optional
import functools

# Global cached Langfuse client (singleton pattern)
_langfuse_client = None


def get_langfuse_client():
    """
    Get Langfuse client instance (singleton pattern).
    Used for tracing, scoring, and custom events.
    
    Returns:
        Langfuse client instance (cached singleton)
    """
    global _langfuse_client
    
    if _langfuse_client is None:
        try:
            # Verify credentials are set
            if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
                return None
            
            _langfuse_client = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST
            )
        except Exception as e:
            return None
    
    return _langfuse_client


# Alias for backward compatibility
get_langfuse_handler = get_langfuse_client


@contextmanager
def trace_agent(client: Optional[Langfuse], agent_name: str, input_data: Dict[str, Any], metadata: Dict[str, Any] = None):
    """
    Context manager to trace an agent execution as a SPAN.
    Thread-safe: Gracefully handles parallel execution contexts.
    
    Args:
        client: Langfuse client instance
        agent_name: Name of the agent
        input_data: Input data for the agent
        metadata: Additional metadata
        
    Yields:
        Client instance (for compatibility)
    """
    if not client:
        yield None
        return
    
    try:
        # Try to use context-based tracing, but catch thread context errors
        with client.start_as_current_span(
            name=agent_name,
            input=input_data,
            metadata=metadata or {}
        ):
            yield client
    except Exception as e:
        # Thread context errors are expected in parallel execution - fail gracefully
        if "context" in str(e).lower() or "thread" in str(e).lower():
            # Just yield client without span context - events will still be logged to trace
            yield client
        else:
            print(f"⚠ Agent span tracing warning for {agent_name}: {str(e)}")
            yield None


@contextmanager
def trace_llm_call(client: Optional[Langfuse], model: str, input_data: Dict[str, Any], metadata: Dict[str, Any] = None):
    """
    Context manager to trace an LLM call as a GENERATION.
    Thread-safe: Gracefully handles parallel execution contexts.
    
    Args:
        client: Langfuse client instance
        model: Model name (e.g., 'gpt-4o-mini')
        input_data: Input data for the generation
        metadata: Additional metadata
        
    Yields:
        Client instance (for compatibility)
    """
    if not client:
        yield None
        return
    
    try:
        # Try to use context-based generation tracking
        with client.start_as_current_generation(
            name=f"{model}-generation",
            input=input_data,
            model=model,
            metadata=metadata or {}
        ):
            yield client
    except Exception as e:
        # Thread context errors are expected in parallel execution - fail gracefully
        if "context" in str(e).lower() or "thread" in str(e).lower():
            # Just yield client - basic tracing will still work
            yield client
        else:
            print(f"⚠ Generation tracing warning: {str(e)}")
            yield None


def log_event(client: Optional[Langfuse], event_name: str, data: Dict[str, Any], metadata: Dict[str, Any] = None):
    """
    Log a decision or event as an EVENT.
    
    Args:
        client: Langfuse client instance
        event_name: Name of the event
        data: Event data
        metadata: Additional metadata
    """
    if not client:
        return
    
    try:
        # Use create_event for proper event logging
        client.create_event(
            name=event_name,
            input=data,
            metadata=metadata or {}
        )
    except Exception as e:
        print(f"⚠ Event logging warning for {event_name}: {str(e)}")


def update_span_output(span_or_client: Any, output: Dict[str, Any]):
    """
    Helper to update span output - works with both span objects and client references.
    Thread-safe helper for updating span/generation outputs.
    
    Args:
        span_or_client: Either a span/generation object or Langfuse client
        output: Output data to log
    """
    if not span_or_client:
        return
    
    try:
        # If it's a span or generation object, update it directly
        if hasattr(span_or_client, 'end'):
            # It's a span/generation object - update will be called on end()
            pass  # Spans are updated when end() is called
        # If it's the client, try to update current span (backward compatibility)
        elif hasattr(span_or_client, 'update_current_span'):
            span_or_client.update_current_span(output=output)
    except Exception as e:
        print(f"⚠ Span update warning: {str(e)}")


def log_scores_to_langfuse(trace_id: str, scores: dict):
    """
    Log evaluation scores to Langfuse.
    
    Args:
        trace_id: Trace identifier from Langfuse
        scores: Dictionary of score names and values
    """
    try:
        client = get_langfuse_client()
        if not client:
            return
        
        for score_name, score_value in scores.items():
            client.score(
                trace_id=trace_id,
                name=score_name,
                value=score_value
            )
        
        print(f"✓ Logged {len(scores)} scores to Langfuse for trace {trace_id}")
    except Exception as e:
        print(f"⚠ Score logging warning: {str(e)}")


