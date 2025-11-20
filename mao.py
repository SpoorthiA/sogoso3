"""Integrated MAO with Langfuse observability and Ragas evaluation."""
from typing import Dict, Any, Optional
import time
import sys
from io import StringIO
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph.graph import StateGraph, END
from state import GraphState
from agents import (
    preprocessing_agent,
    planning_agent,
    product_agent,
    knowledge_agent,
    promotion_agent,
    synthesizer_agent
)
from config import ENABLE_PROMOTIONS_AGENT
from evaluation import evaluate_response, extract_contexts_from_results
from memory import save_to_chat_history


@contextmanager
def suppress_langfuse_errors():
    """Suppress verbose Langfuse error output (HTML error pages, etc.)."""
    old_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr


def parallel_domain_executor(state: GraphState) -> Dict[str, Any]:
    """
    Execute all domain agents in TRUE parallel.
    Each agent is fully traced with individual spans visible in Langfuse.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with results from all domain agents
    """
    client = state.get("langfuse_client")
    plan = state.get("plan", [])
    
    print(f"⚡ Parallel Executor: Running {len(plan)} agents in TRUE parallel...")
    
    # Map agent names to functions (conditionally include promotions)
    agent_map = {
        "product_agent": product_agent,
        "knowledge_agent": knowledge_agent,
    }
    
    # Only include promotion agent if enabled
    if ENABLE_PROMOTIONS_AGENT:
        agent_map["promotion_agent"] = promotion_agent
    
    intermediate_results = state.get("intermediate_results", {})
    start_time = time.time()
    
    # Run agents in separate threads WITHOUT span wrapper (which blocks parallelism)
    results = {}
    errors = {}
    
    def run_single_agent(agent_name, agent_func, agent_state):
        """Run a single agent and return results"""
        try:
            agent_start = time.time()
            result = agent_func(agent_state)
            elapsed = time.time() - agent_start
            return agent_name, result, None, elapsed
        except Exception as e:
            elapsed = time.time() - agent_start
            return agent_name, None, str(e), elapsed
    
    # Use ThreadPoolExecutor for true parallelism
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all agents simultaneously
        future_to_agent = {}
        for agent_name in plan:
            if agent_name in agent_map:
                future = executor.submit(
                    run_single_agent,
                    agent_name,
                    agent_map[agent_name],
                    state  # Pass full state with langfuse client
                )
                future_to_agent[future] = agent_name
        
        # Collect results as they complete
        for future in as_completed(future_to_agent):
            agent_name, result, error, elapsed = future.result()
            
            if error:
                print(f"⚠ {agent_name} failed ({elapsed:.2f}s): {error}")
                errors[agent_name] = error
            else:
                print(f"✓ {agent_name} completed ({elapsed:.2f}s)")
                if result and "intermediate_results" in result:
                    intermediate_results.update(result["intermediate_results"])
                results[agent_name] = result
    
    total_time = time.time() - start_time
    
    # Log Domain-Agents execution as an event
    if client:
        try:
            from observability import log_event
            log_event(
                client=client,
                event_name="Domain-Agents",
                data={
                    "agents": plan,
                    "execution_mode": "parallel",
                    "completed": list(results.keys()),
                    "failed": list(errors.keys()),
                    "total_time_seconds": total_time,
                    "num_agents": len(plan)
                },
                metadata={"parallelization": "ThreadPoolExecutor"}
            )
        except Exception as e:
            print(f"⚠ Domain-Agents logging error: {str(e)}")
    
    print(f"✓ Domain agents complete: {len(results)}/{len(plan)} in {total_time:.2f}s")
    
    return {"intermediate_results": intermediate_results}


def create_mao_graph_with_observability():
    """
    Create MAO graph with Langfuse observability and parallel domain execution.
    
    Returns:
        Compiled LangGraph workflow
    """
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("preprocessing", preprocessing_agent)
    workflow.add_node("planning", planning_agent)
    workflow.add_node("parallel_executor", parallel_domain_executor)
    workflow.add_node("synthesizer", synthesizer_agent)
    
    # Set entry point
    workflow.set_entry_point("preprocessing")
    
    # Add edges - simple linear flow with parallel execution
    workflow.add_edge("preprocessing", "planning")
    workflow.add_edge("planning", "parallel_executor")
    workflow.add_edge("parallel_executor", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()


def run_mao_with_observability(
    query: str,
    session_id: str = "default",
    user_id: Optional[str] = None,
    enable_evaluation: bool = False,
    progress_callback: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Run MAO with full observability, memory, and evaluation.
    
    Args:
        query: User's question
        session_id: Session identifier
        user_id: Optional user identifier
        enable_evaluation: Whether to run Ragas evaluation
        progress_callback: Optional callback for evaluation progress
        
    Returns:
        Dictionary with final_answer and metadata
    """
    overall_start = time.time()
    
    # Create initial state
    initial_state = {
        "user_query": query,
        "session_id": session_id,
        "chat_history": [],  # Empty for performance
        "entities": None,
        "plan": [],
        "subqueries": {},
        "current_agent_index": 0,
        "intermediate_results": {},
        "final_answer": "",
        "error": None,
        "trace_id": None,
        "langfuse_client": None
    }
    
    # Initialize Langfuse client and create trace
    langfuse_client = None
    trace_id = None
    
    try:
        from observability import get_langfuse_client
        langfuse_client = get_langfuse_client()
        
        if langfuse_client:
            # Create trace ID
            trace_id = langfuse_client.create_trace_id()
            initial_state["trace_id"] = trace_id
            initial_state["langfuse_client"] = langfuse_client
    except Exception as e:
        pass  # Silently handle Langfuse errors
    
    # Run graph
    graph = create_mao_graph_with_observability()
    
    try:
        # Use context manager for proper Langfuse span lifecycle
        if langfuse_client:
            with langfuse_client.start_as_current_span(
                name="SOGOSO-Chat",
                input={"query": query},
                metadata={
                    "session_id": session_id,
                    "user_id": user_id or "anonymous",
                    "trace_id": trace_id
                }
            ):
                # Set session ID on the trace for proper session grouping
                langfuse_client.update_current_trace(
                    session_id=session_id,
                    user_id=user_id or "anonymous",
                    tags=["sogoso", "multi-agent", "chat"]
                )
                
                # Execute the graph inside span context
                final_state = graph.invoke(initial_state)
                
                final_answer = final_state.get("final_answer", "")
                entities = final_state.get("entities")
                plan = final_state.get("plan", [])
                intermediate_results = final_state.get("intermediate_results", {})
                
                # Update trace with results
                langfuse_client.update_current_trace(
                    output={"answer": final_answer},
                    metadata={
                        "entities": str(entities),
                        "plan": plan,
                        "agents_called": list(intermediate_results.keys())
                    }
                )
                
                # Save to chat history (async - don't block on database)
                try:
                    save_to_chat_history(session_id, query, final_answer)
                except Exception as e:
                    print(f"⚠ Save warning: {str(e)}")
                
                # Run evaluation if enabled
                evaluation_scores = {}
                if enable_evaluation and final_answer:
                    # Wrap evaluation in a span for proper tracking
                    with langfuse_client.start_as_current_span(
                        name="Evaluation",
                        input={
                            "query": query,
                            "answer": final_answer[:200] + "..." if len(final_answer) > 200 else final_answer
                        },
                        metadata={"evaluation_framework": "ragas"}
                    ):
                        try:
                            contexts = extract_contexts_from_results(intermediate_results)
                            
                            if contexts:
                                evaluation_scores = evaluate_response(
                                    query=query,
                                    answer=final_answer,
                                    contexts=contexts,
                                    progress_callback=progress_callback
                                )
                                
                                # Log scores to Langfuse (within span context)
                                if evaluation_scores:
                                    for score_name, score_value in evaluation_scores.items():
                                        try:
                                            # Convert score to float
                                            if isinstance(score_value, (list, tuple)):
                                                score_val = float(score_value[0]) if score_value else 0.0
                                            else:
                                                score_val = float(score_value)
                                            
                                            # Create score (no trace_id needed in context)
                                            langfuse_client.create_score(
                                                name=score_name,
                                                value=score_val
                                            )
                                        except Exception as e:
                                            pass  # Silently handle score logging errors
                                
                                # Update span with output
                                langfuse_client.update_current_span(
                                    output={
                                        "scores": evaluation_scores,
                                        "num_metrics": len(evaluation_scores)
                                    }
                                )
                        except Exception as e:
                            langfuse_client.update_current_span(
                                output={"error": str(e)},
                                level="ERROR"
                            )
                
                # Flush Langfuse to send traces to server
                if langfuse_client:
                    try:
                        langfuse_client.flush()
                    except Exception as e:
                        pass  # Silently handle flush errors
                
                # Calculate total latency and print at the end
                total_latency = time.time() - overall_start
                print(f"⏱️ Total execution time: {total_latency:.2f}s")
                
                return {
                    "final_answer": final_answer,
                    "intermediate_results": intermediate_results,
                    "entities": entities,
                    "plan": plan,
                    "error": final_state.get("error"),
                    "trace_id": trace_id,
                    "evaluation_scores": evaluation_scores,
                    "latency": total_latency
                }
        else:
            # No Langfuse client - run without tracing
            final_state = graph.invoke(initial_state)
            
            final_answer = final_state.get("final_answer", "")
            entities = final_state.get("entities")
            plan = final_state.get("plan", [])
            intermediate_results = final_state.get("intermediate_results", {})
            
            save_to_chat_history(session_id, query, final_answer)
            
            evaluation_scores = {}
            if enable_evaluation and final_answer:
                try:
                    contexts = extract_contexts_from_results(intermediate_results)
                    if contexts:
                        evaluation_scores = evaluate_response(
                            query=query,
                            answer=final_answer,
                            contexts=contexts,
                            progress_callback=progress_callback
                        )
                except Exception as e:
                    pass  # Silently handle evaluation errors
            
            # Calculate total latency and print at the end
            total_latency = time.time() - overall_start
            print(f"⏱️ Total execution time: {total_latency:.2f}s")
            
            return {
                "final_answer": final_answer,
                "intermediate_results": intermediate_results,
                "entities": entities,
                "plan": plan,
                "error": final_state.get("error"),
                "trace_id": None,
                "evaluation_scores": evaluation_scores,
                "latency": total_latency
            }
    except Exception as e:
        print(f"✗ Error in MAO execution: {str(e)}")
        
        # Log error to Langfuse with context manager
        if langfuse_client:
            try:
                with langfuse_client.start_as_current_span(
                    name="SOGOSO-Chat-Error",
                    input={"query": query},
                    metadata={"trace_id": trace_id, "error": str(e)}
                ):
                    langfuse_client.update_current_trace(
                        output={"error": str(e)},
                        level="ERROR"
                    )
                    with suppress_langfuse_errors():
                        langfuse_client.flush()
            except:
                pass
        
        # Calculate latency even on error and print
        total_latency = time.time() - overall_start
        print(f"⏱️ Total execution time: {total_latency:.2f}s")
        
        return {
            "final_answer": "I apologize, but I encountered an error. Please try again.",
            "error": str(e),
            "intermediate_results": {},
            "entities": None,
            "plan": [],
            "trace_id": trace_id,
            "evaluation_scores": {},
            "latency": total_latency
        }


if __name__ == "__main__":
    # Test with observability
    print("=" * 60)
    print("SOGOSO MAO Test with Observability")
    print("=" * 60)
    
    result = run_mao_with_observability(
        query="Show me composite baseball bats under $250",
        session_id="test_session",
        enable_evaluation=True
    )
    
    print(f"\n{'='*60}")
    print("FINAL ANSWER:")
    print(f"{'='*60}")
    print(result["final_answer"])
    
    if result.get("evaluation_scores"):
        print(f"\n{'='*60}")
        print("EVALUATION SCORES:")
        print(f"{'='*60}")
        for metric, score in result["evaluation_scores"].items():
            print(f"{metric}: {score:.4f}")
