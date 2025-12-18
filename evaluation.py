"""Ragas evaluation integration."""
import warnings
warnings.filterwarnings("ignore")  # Suppress all evaluation warnings

from typing import Dict, List, Optional, Callable
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
try:
    from ragas.metrics import context_utilization
except ImportError:
    try:
        from ragas.metrics import context_relevancy as context_utilization
    except ImportError:
        context_utilization = None

try:
    from ragas.metrics.critique import AspectCritique
except ImportError:
    AspectCritique = None

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import OPENAI_API_KEY, OPENAI_MODEL


def evaluate_response(
    query: str,
    answer: str,
    contexts: List[str],
    ground_truth: str = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None
) -> Dict[str, float]:
    """
    Evaluate response quality using Ragas metrics with progress tracking.
    
    Args:
        query: User's question
        answer: Generated answer
        contexts: Retrieved context documents
        ground_truth: Optional ground truth answer
        progress_callback: Optional callback function(metric_name, current, total)
        
    Returns:
        Dictionary of metric scores
    """
    if progress_callback:
        progress_callback("Initializing evaluation...", 0, 100)
    
    # Prepare data for Ragas (using standard column names for v0.2+)
    data = {
        "question": [query],
        "answer": [answer],
        "contexts": [contexts],
    }
    
    if ground_truth:
        data["ground_truth"] = [ground_truth]
    
    dataset = Dataset.from_dict(data)
    
    # Define metrics
    metrics_to_run = [
        ("Faithfulness", faithfulness),
        ("Answer Relevancy", answer_relevancy),
    ]
    
    # Add Context Utilization if available
    if context_utilization:
        metrics_to_run.append(("Context Utilization", context_utilization))
        
    # Add Aspect Critique (Professionalism) if available
    if AspectCritique:
        professionalism = AspectCritique(
            name="Professionalism",
            definition="The answer should be helpful, polite, and professional.",
            strictness=3
        )
        metrics_to_run.append(("Professionalism", professionalism))
    
    # Run evaluation with optimized settings
    try:
        if progress_callback:
            progress_callback("Running evaluation metrics...", 10, 100)
        
        result = evaluate(
            dataset,
            metrics=[m[1] for m in metrics_to_run],
            llm=ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0),
            embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        )
        
        if progress_callback:
            progress_callback("Processing results...", 90, 100)
        
        # Extract scores
        scores = {}
        
        # Handle different result formats
        if hasattr(result, 'to_pandas'):
            # Result is a Ragas Result object - convert to pandas first
            df = result.to_pandas()
            
            for display_name, metric_obj in metrics_to_run:
                metric_key = display_name.lower().replace(" ", "_")
                if metric_key in df.columns:
                    value = df[metric_key].iloc[0]
                    if value is not None and not (isinstance(value, float) and value != value):  # Check for NaN
                        scores[display_name] = float(value)
        elif isinstance(result, dict):
            # Result is already a dictionary
            for display_name, metric_obj in metrics_to_run:
                metric_key = display_name.lower().replace(" ", "_")
                if metric_key in result:
                    value = result[metric_key]
                    # Handle both list and scalar returns
                    if isinstance(value, list):
                        scores[display_name] = float(value[0]) if value else 0.0
                    else:
                        scores[display_name] = float(value)
        
        if progress_callback:
            progress_callback("Evaluation complete!", 100, 100)
        
        print(f"✓ Ragas evaluation complete with {len(scores)} metrics")
        return scores
    except Exception as e:
        print(f"✗ Error in Ragas evaluation: {str(e)}")
        if progress_callback:
            progress_callback(f"Error: {str(e)}", 100, 100)
        return {}


def extract_contexts_from_results(intermediate_results: Dict) -> List[str]:
    """
    Extract context strings from intermediate agent results.
    
    Args:
        intermediate_results: Results from domain agents
        
    Returns:
        List of context strings
    """
    contexts = []
    
    # Extract from product agent
    if "product_agent" in intermediate_results:
        products = intermediate_results["product_agent"].get("data", [])
        for p in products:
            contexts.append(p.get("description", ""))
    
    # Extract from knowledge agent
    if "knowledge_agent" in intermediate_results:
        policies = intermediate_results["knowledge_agent"].get("data", [])
        for pol in policies:
            contexts.append(pol.get("content", ""))
    
    # Extract from promotion agent
    if "promotion_agent" in intermediate_results:
        promos = intermediate_results["promotion_agent"].get("data", [])
        for promo in promos:
            contexts.append(promo.get("content", ""))
    
    # Filter out empty contexts
    contexts = [c for c in contexts if c.strip()]
    
    return contexts
