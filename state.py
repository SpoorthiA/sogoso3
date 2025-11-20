"""State definitions for the MAO (Multi-Agent Orchestration) graph."""
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ExtractedEntities(BaseModel):
    """Structured entities extracted from user query."""
    product: Optional[str] = Field(None, description="Product type or category mentioned (e.g., 'bat', 'glove', 'helmet')")
    brand: Optional[str] = Field(None, description="Brand name if specified")
    intent: Optional[str] = Field(None, description="User intent (e.g., 'purchase', 'return_policy', 'promotion', 'compare')")
    price_min: Optional[float] = Field(None, description="Minimum price filter")
    price_max: Optional[float] = Field(None, description="Maximum price filter")
    material: Optional[str] = Field(None, description="Material type (e.g., 'wood', 'aluminum', 'composite')")
    category: Optional[str] = Field(None, description="Product category")
    keywords: List[str] = Field(default_factory=list, description="Additional search keywords")


class GraphState(TypedDict):
    """State that flows through the MAO graph."""
    # Input
    user_query: str
    session_id: str
    
    # Chat History
    chat_history: List[Dict[str, str]]
    
    # Processing Results
    entities: Optional[ExtractedEntities]
    plan: List[str]  # List of agent names to call
    subqueries: Dict[str, str]  # Subqueries for each agent {agent_name: subquery}
    current_agent_index: int  # Track which agent in plan to execute next
    
    # Domain Agent Results
    intermediate_results: Dict[str, Any]  # Results from domain agents
    
    # Final Output
    final_answer: str
    
    # Metadata
    error: Optional[str]
    trace_id: Optional[str]
    langfuse_client: Optional[Any]  # Langfuse client for tracing


class DomainAgentResult(BaseModel):
    """Result from a domain agent."""
    agent_name: str
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
