"""Agent implementations for SOGOSO."""
import os
import warnings

# Disable ChromaDB telemetry BEFORE importing chromadb
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import Dict, Any, List, Callable
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from state import GraphState, ExtractedEntities, DomainAgentResult
from chroma_pool import get_chroma_client
from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    CHROMA_PERSIST_DIRECTORY,
    EMBEDDING_MODEL,
    MAX_TOKENS,
    REQUEST_TIMEOUT,
    ENABLE_PROMOTIONS_AGENT
)
from evaluation import evaluate_response, extract_contexts_from_results
from observability import trace_agent, trace_llm_call, log_event


# Initialize ChatOpenAI model with performance optimizations
llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0,
    max_tokens=MAX_TOKENS,
    request_timeout=REQUEST_TIMEOUT
)

# Cache prompt templates for better performance (avoid recreating them)
_PREPROCESSING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at extracting structured information from user queries about baseball equipment.
Extract relevant entities such as:
- product: type of product (bat, glove, helmet, etc.)
- brand: brand name if mentioned
- intent: what the user wants (purchase, return_policy, promotion, compare, etc.)
- price_min/price_max: price range filters
- material: material type (wood, aluminum, composite, leather, etc.)
- category: product category
- keywords: additional search terms

Be precise and only extract entities that are explicitly mentioned or clearly implied."""),
    ("user", "{query}")
])

_PLANNING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a planning agent for a sports goods shop assistant. Determine which agents to call AND generate focused subqueries for each.

Available agents:
- product_agent: Product info, prices, availability. Use for product searches, comparisons, recommendations.
- knowledge_agent: Policies, SOPs, returns, warranties. Use for policy/return questions.

Return a JSON object with two keys:
1. "plan": array of agent names to call in order
2. "subqueries": object mapping each agent to its focused search query

Guidelines for subqueries:
- product_agent: Focus on features, specs, price ranges, materials, brands
- knowledge_agent: Focus on policies, procedures, rules, returns, warranties

Example output:
{{
  "plan": ["product_agent", "knowledge_agent"],
  "subqueries": {{
    "product_agent": "composite baseball bats under $200",
    "knowledge_agent": "return policy for baseball equipment"
  }}
}}

NOTE: Only include agents in the plan if they are needed for the query."""),
    ("user", """User query: {query}
Entities: {entities}

Return JSON with 'plan' and 'subqueries':""")
])

_SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a professional sports equipment shop assistant. Create clear, well-formatted responses using proper markdown.

CRITICAL FORMATTING RULES:
- Use markdown bold: **Product Name** (double asterisks, no spaces inside)
- Use dash bullets: - Item (NOT • or other symbols)
- Show prices clearly: $299.99
- Use line breaks between items
- Be thorough but concise
- Be friendly and professional
- Always complete your sentences and thoughts

EXAMPLE OUTPUT:
We have excellent composite bats available:

- **DiamondMax Composite Bat** by Rawlings - $299.99
- **TechSwing Smart Bat** by Easton - $399.99
- **VelocityMax Hybrid Bat** by Marucci - $219.99

Let me know if you need more information!

DO NOT use unicode symbols, weird spacing, or special characters. Keep formatting simple and clean."""),
    ("user", """Query: {query}

{context}

Provide a helpful, well-formatted response:""")
])


def _parse_json_response(response_content: str) -> Any:
    """
    Parse JSON from LLM response, handling markdown code blocks.
    
    Args:
        response_content: Raw response content from LLM
        
    Returns:
        Parsed JSON object
    """
    content = response_content.strip()
    
    # Remove markdown code blocks if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    
    return json.loads(content.strip())


def _format_products_context(products: List[Dict]) -> List[str]:
    """Format product data for synthesizer context."""
    lines = ["PRODUCTS AVAILABLE:"]
    for p in products[:3]:  # Limit to top 3
        name = p.get('product_name', p.get('name', 'Unknown'))
        price = p.get('price', 0)
        brand = p.get('brand', '')
        lines.append(f"- {name} by {brand}: ${price}")
    return lines


def _format_policies_context(policies: List[Dict]) -> List[str]:
    """Format policy data for synthesizer context."""
    lines = ["\nPOLICIES:"]
    for pol in policies[:2]:  # Limit to top 2
        title = pol.get('title', 'Policy')
        content = pol.get('content', '')[:150]  # Truncate long content
        lines.append(f"- {title}: {content}")
    return lines


def _format_promotions_context(promotions: List[Dict]) -> List[str]:
    """Format promotion data for synthesizer context."""
    lines = ["\nACTIVE PROMOTIONS:"]
    for promo in promotions[:3]:  # Limit to top 3
        name = promo.get('name', 'Promotion')
        content = promo.get('content', '')[:150]  # Truncate long content
        lines.append(f"- {name}: {content}")
    return lines


def _handle_agent_error(client, error: Exception, agent_name: str) -> Dict[str, Any]:
    """
    Standardized error handling for agents.
    
    Args:
        client: Langfuse client for logging
        error: Exception that occurred
        agent_name: Name of the agent where error occurred
        
    Returns:
        Error state dictionary
    """
    error_msg = str(error)
    print(f"✗ Error in {agent_name}: {error_msg}")
    
    if client:
        client.update_current_span(
            output={"error": error_msg},
            level="ERROR"
        )
    
    return {"error": error_msg}


def _print_retrieved_chunks(chunk_type: str, items: List[Dict], format_func: Callable[[Dict], str]) -> None:
    """
    Print retrieved chunks in a consistent format.
    
    Args:
        chunk_type: Type of chunks (e.g., "product", "knowledge", "promotion")
        items: List of retrieved items
        format_func: Function to format each item for display
    """
    chunk_output = [f"✓ Retrieved {len(items)} {chunk_type} chunks:"]
    for i, item in enumerate(items, 1):
        chunk_output.append(f"  [{i}] {format_func(item)}")
    chunk_output.append(f"✓ Found {len(items)} {chunk_type}s" if chunk_type != "knowledge" else f"✓ Found {len(items)} relevant policies")
    print("\n".join(chunk_output))


def preprocessing_agent(state: GraphState) -> Dict[str, Any]:
    """
    Extract structured entities from user query using OpenAI function calling.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with extracted entities
    """
    print("STEP: Analyzing User Intent | Identifying key entities and intent")
    
    client = state.get("langfuse_client")
    
    with trace_agent(client, "Preprocessing-Agent", {"query": state["user_query"]}, {"step": "entity_extraction"}):
        # Bind the structured output schema
        structured_llm = llm.with_structured_output(ExtractedEntities)
        
        # Use cached prompt template
        chain = _PREPROCESSING_PROMPT | structured_llm
        
        # Extract entities
        try:
            with trace_llm_call(client, OPENAI_MODEL, {"messages": [{"role": "user", "content": state["user_query"]}]}, {"task": "entity_extraction"}):
                entities = chain.invoke({
                    "query": state["user_query"],
                })
            
            # Glass box output
            if entities.intent:
                print(f"GLASS: Intent | {entities.intent}")
            if entities.product:
                print(f"GLASS: Product | {entities.product}")
            if entities.brand:
                print(f"GLASS: Brand | {entities.brand}")
            if entities.price_max:
                print(f"GLASS: Max Price | ${entities.price_max}")
            
            # Update span with output
            if client:
                client.update_current_span(
                    output={"entities": entities.model_dump()}
                )
            
            return {"entities": entities}
        except Exception as e:
            return {**_handle_agent_error(client, e, "preprocessing"), "entities": None}


def planning_agent(state: GraphState) -> Dict[str, Any]:
    """
    Analyze entities and create execution plan with subqueries for each agent.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with agent execution plan and subqueries
    """
    print("STEP: Planning Execution | Determining necessary agents and subqueries")
    
    client = state.get("langfuse_client")
    entities = state.get("entities")
    
    if not entities:
        return {"plan": [], "subqueries": {}, "error": "No entities extracted"}
    
    with trace_agent(client, "Planning-Agent", {"entities": entities.model_dump() if entities else {}}, {"step": "agent_planning"}):
        # Use cached combined prompt template
        combined_chain = _PLANNING_PROMPT | llm
        
        try:
            # Single LLM call for both plan and subqueries (faster!)
            with trace_llm_call(client, OPENAI_MODEL, {"query": state['user_query'], "entities": entities.model_dump()}, {"task": "combined_planning"}):
                response = combined_chain.invoke({
                    "query": state["user_query"],
                    "entities": str(entities.model_dump() if entities else {})
                })
            
            # Parse the combined response
            result = _parse_json_response(response.content)
            plan = result.get("plan", [])
            subqueries = result.get("subqueries", {})
            
            # Glass box output
            print(f"GLASS: Plan | {', '.join(plan)}")
            for agent, query in subqueries.items():
                print(f"GLASS: Subquery ({agent}) | {query}")
            
            # Log event for plan creation with subqueries
            log_event(client, "AgentPlanCreated", {
                "plan": plan, 
                "num_agents": len(plan),
                "subqueries": subqueries
            }, {"decision_type": "agent_routing"})
            
            # Update span with output
            if client:
                client.update_current_span(
                    output={"plan": plan, "subqueries": subqueries}
                )
            
            return {"plan": plan, "subqueries": subqueries, "current_agent_index": 0}
        except Exception as e:
            print(f"✗ Error in planning: {str(e)}")
            # Default fallback plan
            if entities and entities.intent:
                if "policy" in entities.intent.lower() or "return" in entities.intent.lower():
                    plan = ["knowledge_agent"]
                elif "promotion" in entities.intent.lower() or "discount" in entities.intent.lower():
                    plan = ["promotion_agent", "product_agent"]
                else:
                    plan = ["product_agent"]
            else:
                plan = ["product_agent"]
            
            # Generate simple subqueries for fallback
            subqueries = {}
            for agent in plan:
                subqueries[agent] = state["user_query"]  # Use original query as fallback
            
            # Print fallback plan and subqueries together
            fallback_output = [f"⚠ Using fallback plan: {plan}"]
            fallback_output.append("✓ Generated subqueries:")
            for agent_name, subquery in subqueries.items():
                fallback_output.append(f"  • {agent_name}: {subquery}")
            print("\n".join(fallback_output))
            
            if client:
                client.update_current_span(
                    output={"plan": plan, "subqueries": subqueries, "fallback": True, "error": str(e)},
                    level="WARNING"
                )
            
            return {"plan": plan, "subqueries": subqueries, "current_agent_index": 0}


def product_agent(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve products from ChromaDB with semantic search and metadata filtering.
    
    Args:
        state: Current graph state
        
    Returns:
        Product search results
    """
    print("STEP: Searching Products | Querying vector database for products")
    
    client = state.get("langfuse_client")
    entities = state.get("entities")
    subqueries = state.get("subqueries", {})
    
    if not entities:
        return {"error": "No entities for product search"}
    
    # Get subquery for this agent if available
    agent_subquery = subqueries.get("product_agent")
    
    with trace_agent(client, "Product-Agent", {
        "entities": entities.model_dump(),
        "subquery": agent_subquery
    }, {"agent_type": "domain", "collection": "products"}):
        # Use pooled ChromaDB connection (reuses embeddings and client)
        db = get_chroma_client("products")
        
        # Use subquery if available, otherwise build from entities
        if agent_subquery:
            query = agent_subquery
        else:
            # Build search query from entities (fallback)
            search_terms = []
            if entities.product:
                search_terms.append(entities.product)
            if entities.brand:
                search_terms.append(entities.brand)
            if entities.material:
                search_terms.append(entities.material)
            search_terms.extend(entities.keywords)
            
            query = " ".join(search_terms) if search_terms else state["user_query"]
        
        # Build metadata filter
        filter_dict = {}
        if entities.price_max:
            filter_dict["price"] = {"$lte": entities.price_max}
        if entities.price_min:
            if "price" in filter_dict:
                filter_dict["price"]["$gte"] = entities.price_min
            else:
                filter_dict["price"] = {"$gte": entities.price_min}
        
        # Glass box output
        print(f"GLASS: Query | {query}")
        if filter_dict:
            print(f"GLASS: Filters | {filter_dict}")
        
        # Perform search (k=2 for maximum speed)
        try:
            if filter_dict:
                results = db.similarity_search(query, k=2, filter=filter_dict)
            else:
                results = db.similarity_search(query, k=2)
            
            # Format results
            products = []
            for doc in results:
                products.append({
                    "product_name": doc.metadata.get("product_name"),
                    "category": doc.metadata.get("category"),
                    "brand": doc.metadata.get("brand"),
                    "price": doc.metadata.get("price"),
                    "title": doc.metadata.get("title"),
                    "review_rating": doc.metadata.get("review_rating"),
                    "review_count": doc.metadata.get("review_count"),
                    "description": doc.page_content
                })
            
            # Log search event with results
            log_event(client, "VectorSearch", {
                "collection": "products",
                "query": agent_subquery or query,  # Use subquery if available, otherwise query
                "filters": filter_dict, 
                "k": 3,
                "results_count": len(products),
                "retrieved_products": [
                    {
                        "product_name": p["product_name"],
                        "brand": p["brand"],
                        "price": p["price"],
                        "description": p["description"][:100]  # Truncate for readability
                    } for p in products
                ]
            }, {"search_type": "similarity", "agent": "product"})
            
            print(f"INFO: Found {len(products)} products")
            
            result = DomainAgentResult(
                agent_name="product_agent",
                success=True,
                data=products,
                metadata={"count": len(products), "query": query}
            )
            
            # Update trace
            if client:
                client.update_current_trace(
                    output={"products": [p["name"] for p in products], "count": len(products)}
                )
            
            intermediate_results = state.get("intermediate_results", {})
            intermediate_results["product_agent"] = result.model_dump()
            
            return {"intermediate_results": intermediate_results}
        except Exception as e:
            return _handle_agent_error(client, e, "product search")


def knowledge_agent(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve policies from ChromaDB.
    
    Args:
        state: Current graph state
        
    Returns:
        Policy/knowledge search results
    """
    print("STEP: Checking Policies | Querying knowledge base for policies")
    
    client = state.get("langfuse_client")
    subqueries = state.get("subqueries", {})
    
    # Get subquery for this agent if available
    agent_subquery = subqueries.get("knowledge_agent")
    
    with trace_agent(client, "Knowledge-Agent", {
        "query": state["user_query"],
        "subquery": agent_subquery
    }, {"agent_type": "domain", "collection": "knowledge"}):
        # Use pooled ChromaDB connection (reuses embeddings and client)
        db = get_chroma_client("knowledge")
        
        # Use subquery if available, otherwise use original query
        query = agent_subquery if agent_subquery else state["user_query"]
        
        # Glass box output
        print(f"GLASS: Query | {query}")
        
        # Perform search (k=1 for maximum speed - only best match)
        try:
            results = db.similarity_search(query, k=1)
            
            # Format results
            policies = []
            for doc in results:
                policies.append({
                    "policy_id": doc.metadata.get("policy_id"),
                    "title": doc.metadata.get("title"),
                    "applies_to": doc.metadata.get("applies_to"),
                    "content": doc.page_content
                })
            
            # Log search event with results
            log_event(client, "VectorSearch", {
                "collection": "knowledge",
                "query": agent_subquery or state["user_query"],
                "k": 2,
                "results_count": len(policies),
                "retrieved_policies": [
                    {
                        "policy_id": p["policy_id"],
                        "title": p["title"],
                        "applies_to": p["applies_to"],
                        "content": p["content"][:100]  # Truncate for readability
                    } for p in policies
                ]
            }, {"search_type": "similarity", "agent": "knowledge"})
            
            print(f"INFO: Found {len(policies)} policies")
            
            result = DomainAgentResult(
                agent_name="knowledge_agent",
                success=True,
                data=policies,
                metadata={"count": len(policies)}
            )
            
            # Update span with output
            if client:
                client.update_current_span(
                    output={"policies": [p["title"] for p in policies], "count": len(policies)}
                )
            
            intermediate_results = state.get("intermediate_results", {})
            intermediate_results["knowledge_agent"] = result.model_dump()
            
            return {"intermediate_results": intermediate_results}
        except Exception as e:
            return _handle_agent_error(client, e, "knowledge search")


def promotion_agent(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve promotions from ChromaDB.
    
    Args:
        state: Current graph state
        
    Returns:
        Promotion search results
    """
    print("STEP: Checking Promotions | Searching for active promotions")
    
    client = state.get("langfuse_client")
    entities = state.get("entities")
    subqueries = state.get("subqueries", {})
    
    # Get subquery for this agent if available
    agent_subquery = subqueries.get("promotion_agent")
    
    # Use subquery if available, otherwise build from entities
    if agent_subquery:
        query = agent_subquery
    else:
        # Fallback: build search query from entities
        query = state["user_query"]
        if entities and entities.product:
            query = f"{entities.product} promotions discounts"
    
    with trace_agent(client, "Promotion-Agent", {
        "query": query, 
        "subquery": agent_subquery,
        "entities": entities.model_dump() if entities else {}
    }, {"agent_type": "domain", "collection": "promotions"}):
        # Use pooled ChromaDB connection (reuses embeddings and client)
        db = get_chroma_client("promotions")
        
        # Glass box output
        print(f"GLASS: Query | {query}")
        
        # Perform search (k=2 for maximum speed)
        try:
            results = db.similarity_search(query, k=2)
            
            # Format results
            promotions = []
            for doc in results:
                promotions.append({
                    "promo_id": doc.metadata.get("promo_id"),
                    "name": doc.metadata.get("name"),
                    "type": doc.metadata.get("type"),
                    "categories": doc.metadata.get("applicable_categories"),
                    "validity": doc.metadata.get("validity"),
                    "content": doc.page_content
                })
            
            # Log search event with results
            log_event(client, "VectorSearch", {
                "collection": "promotions",
                "query": agent_subquery or query,
                "k": 3,
                "results_count": len(promotions),
                "retrieved_promotions": [
                    {
                        "promo_id": p["promo_id"],
                        "name": p["name"],
                        "type": p["type"],
                        "categories": p["categories"],
                        "validity": p["validity"],
                        "content": p["content"][:100]  # Truncate for readability
                    } for p in promotions
                ]
            }, {"search_type": "similarity", "agent": "promotion"})
            
            print(f"INFO: Found {len(promotions)} promotions")
            
            result = DomainAgentResult(
                agent_name="promotion_agent",
                success=True,
                data=promotions,
                metadata={"count": len(promotions)}
            )
            
            # Update span with output
            if client:
                client.update_current_span(
                    output={"promotions": [p["name"] for p in promotions], "count": len(promotions)}
                )
            
            intermediate_results = state.get("intermediate_results", {})
            intermediate_results["promotion_agent"] = result.model_dump()
            
            return {"intermediate_results": intermediate_results}
        except Exception as e:
            return _handle_agent_error(client, e, "promotion search")


def executor_agent(state: GraphState) -> str:
    """
    Router that determines the next agent to call based on the plan.
    
    Args:
        state: Current graph state
        
    Returns:
        Name of next agent to call, or "synthesizer" if plan is complete
    """
    client = state.get("langfuse_client")
    plan = state.get("plan", [])
    current_index = state.get("current_agent_index", 0)
    
    with trace_agent(client, "Executor-Agent", {"plan": plan, "current_index": current_index}, {"step": "routing"}):
        if current_index < len(plan):
            next_agent = plan[current_index]
            print(f"🔀 Executor: Routing to {next_agent} (step {current_index + 1}/{len(plan)})")
            
            # Log routing decision as event
            log_event(client, "AgentRouting", {
                "next_agent": next_agent,
                "step": current_index + 1,
                "total_steps": len(plan)
            }, {"decision_type": "routing"})
            
            # Update span with output
            if client:
                client.update_current_span(
                    output={"next_agent": next_agent, "step": current_index + 1}
                )
            
            return next_agent
        else:
            print("🔀 Executor: Plan complete, routing to synthesizer")
            
            # Log completion event
            log_event(client, "PlanCompleted", {
                "total_agents_executed": len(plan),
                "routing_to": "synthesizer"
            }, {"decision_type": "completion"})
            
            # Update span with output
            if client:
                client.update_current_span(
                    output={"next_agent": "synthesizer", "plan_completed": True}
                )
            
            return "synthesizer"


def synthesizer_agent(state: GraphState) -> Dict[str, Any]:
    """
    Synthesize final answer from all intermediate results.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with final answer
    """
    print("STEP: Generating Response | Synthesizing information into a natural language answer")
    
    client = state.get("langfuse_client")
    intermediate_results = state.get("intermediate_results", {})
    
    with trace_agent(client, "Synthesizer-Agent", {"num_sources": len(intermediate_results)}, {"step": "answer_synthesis"}):
        # Format context from intermediate results using helper functions
        context_parts = []
        
        if "product_agent" in intermediate_results:
            products = intermediate_results["product_agent"].get("data", [])
            if products:
                context_parts.extend(_format_products_context(products))
        
        if "knowledge_agent" in intermediate_results:
            policies = intermediate_results["knowledge_agent"].get("data", [])
            if policies:
                context_parts.extend(_format_policies_context(policies))
        
        if "promotion_agent" in intermediate_results:
            promos = intermediate_results["promotion_agent"].get("data", [])
            if promos:
                context_parts.extend(_format_promotions_context(promos))
        
        context = "\n".join(context_parts) if context_parts else "No information found."
        
        # Use cached prompt template
        chain = _SYNTHESIZER_PROMPT | llm
        
        try:
            with trace_llm_call(client, OPENAI_MODEL, {"query": state['user_query'], "context_length": len(context)}, {"task": "answer_synthesis"}):
                response = chain.invoke({
                    "query": state["user_query"],
                    "context": context
                })
            
            final_answer = response.content
            print(f"INFO: Generated response ({len(final_answer)} chars)")
            
            # Update span with output
            if client:
                client.update_current_span(
                    output={"answer": final_answer, "answer_length": len(final_answer)}
                )
            
            return {"final_answer": final_answer}
        except Exception as e:
            error_result = _handle_agent_error(client, e, "synthesis")
            return {
                **error_result,
                "final_answer": "I apologize, but I encountered an error processing your request. Please try again."
            }


def increment_agent_index(state: GraphState) -> Dict[str, Any]:
    """Helper function to increment the current agent index."""
    current_index = state.get("current_agent_index", 0)
    return {"current_agent_index": current_index + 1}


def evaluation_node(state: GraphState) -> Dict[str, Any]:
    """
    Evaluate the quality of the generated response.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with evaluation scores
    """
    print("STEP: Quality Evaluation | Assessing response faithfulness and relevance")
    
    client = state.get("langfuse_client")
    query = state.get("user_query")
    final_answer = state.get("final_answer")
    intermediate_results = state.get("intermediate_results", {})
    
    # Skip if no answer generated
    if not final_answer:
        return {"evaluation_scores": {}}
        
    with trace_agent(client, "Evaluation-Node", {}, {"step": "evaluation"}):
        try:
            contexts = extract_contexts_from_results(intermediate_results)
            
            if not contexts:
                print("WARN: No contexts available for evaluation")
                return {"evaluation_scores": {}}
            
            # Run evaluation
            scores = evaluate_response(
                query=query,
                answer=final_answer,
                contexts=contexts
            )
            
            print(f"INFO: Evaluation scores: {scores}")
            
            # Update span with output
            if client:
                client.update_current_span(
                    output={"scores": scores}
                )
                
                # Log scores as separate metrics
                for metric, score in scores.items():
                    try:
                        client.create_score(
                            name=metric,
                            value=score
                        )
                    except:
                        pass
            
            return {"evaluation_scores": scores}
            
        except Exception as e:
            print(f"✗ Error in evaluation node: {str(e)}")
            if client:
                client.update_current_span(
                    output={"error": str(e)},
                    level="ERROR"
                )
            return {"evaluation_scores": {}}
