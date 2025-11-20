"""Streamlit UI for SOGOSO chatbot with live agent thought process."""
import streamlit as st
import uuid
import re
import sys
import time
from typing import Dict, Any
from mao import run_mao_with_observability
from memory import clear_chat_history

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="SOGOSO - Sports Goods Shop Assistant",
    page_icon="⚾",
    layout="wide"
)

# CRITICAL: Pre-initialize ChromaDB connections at app startup (before any requests)
# This prevents the 70+ second initialization delay on first query
@st.cache_resource
def initialize_chroma_connections():
    """Initialize all ChromaDB connections once at app startup."""
    print("🚀 Pre-initializing ChromaDB connections...")
    start = time.time()
    from chroma_pool import get_chroma_client
    from config import ENABLE_PROMOTIONS_AGENT
    
    # Pre-load collections based on feature flags
    get_chroma_client("products")
    get_chroma_client("knowledge")
    if ENABLE_PROMOTIONS_AGENT:
        get_chroma_client("promotions")
    
    elapsed = time.time() - start
    print(f"✓ ChromaDB connections initialized in {elapsed:.2f}s")
    return True

# Initialize connections immediately after page config
_ = initialize_chroma_connections()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .agent-step {
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        border-radius: 6px;
        background-color: #f8f9fa;
        font-family: inherit;
        font-size: 0.9rem;
        color: #333;
        border-left: 3px solid #1f77b4;
        line-height: 1.8;
    }
    .metric-card {
        background: white;
        color: #333;
        padding: 0.8rem;
        border-radius: 6px;
        text-align: left;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #e8e8e8;
        margin: 0.4rem 0;
    }
    .metric-name {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        color: #666;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0.3rem 0;
        color: #1f77b4;
    }
    .overall-score {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
        margin: 1rem 0;
    }
    .overall-score-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .overall-score-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "user_id" not in st.session_state:
        # Generate a persistent user ID (stays same across sessions)
        if "persistent_user_id" not in st.session_state:
            st.session_state.persistent_user_id = str(uuid.uuid4())
        st.session_state.user_id = st.session_state.persistent_user_id
    
    if "messages" not in st.session_state:
        st.session_state.messages = []


def display_metrics(scores: Dict[str, float]):
    """Display evaluation metrics in compact white boxes with overall score."""
    if not scores:
        return
    
    # Calculate overall score
    overall_score = sum(scores.values()) / len(scores) if scores else 0.0
    
    st.markdown("---")
    
    # Display overall score with progress bar
    st.markdown(f"""
    <div class="overall-score">
        <div class="overall-score-label">📊 Overall Quality Score</div>
        <div class="overall-score-value">{overall_score * 100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar for overall score
    st.progress(overall_score)
    
    st.markdown("#### Individual Metrics")
    
    # Metric info
    metric_info = {
        "Faithfulness": "✓",
        "Answer Relevancy": "🎯",
        "Answer Similarity": "🔗",
        "Context Entity Recall": "📚",
        "Answer Correctness": "✅",
        "Context Precision": "🔍",
        "Context Recall": "📈"
    }
    
    # Create 2 columns for compact display
    num_metrics = len(scores)
    cols_per_row = 2
    
    items = list(scores.items())
    for row_start in range(0, num_metrics, cols_per_row):
        cols = st.columns(min(cols_per_row, num_metrics - row_start))
        
        for i, (metric, score) in enumerate(items[row_start:row_start + cols_per_row]):
            # Handle if score is a list or other format
            if isinstance(score, (list, tuple)):
                score = float(score[0]) if score else 0.0
            else:
                score = float(score)
            
            icon = metric_info.get(metric, "📈")
            
            with cols[i]:
                # Format score as percentage
                score_pct = f"{score * 100:.1f}%"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-name">{icon} {metric}</div>
                    <div class="metric-value">{score_pct}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add progress bar below each metric
                st.progress(score)


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">⚾ SOGOSO Sports Goods Shop</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your AI-Powered Baseball Equipment Assistant</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Session Info")
        st.text(f"Session ID: {st.session_state.session_id[:8]}...")
        
        st.markdown("---")
        
        # Settings
        st.header("⚙️ Settings")
        enable_evaluation = st.checkbox("Enable Quality Evaluation", value=False, help="Run Ragas evaluation on responses")
        show_agent_thoughts = st.checkbox("Show Agent Thoughts", value=True, help="Display the multi-agent reasoning process")
        
        st.markdown("---")
        
        # Session controls
        st.header("🔧 Session Controls")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            clear_chat_history(st.session_state.session_id)
            st.success("Chat history cleared!")
            st.rerun()
        
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.success("New session started!")
            st.rerun()
        
        st.markdown("---")
        
        # Example queries
        st.header("💡 Example Queries")
        example_queries = [
            "Show me baseball bats under $200",
            "What's your return policy for gloves?",
            "Are there any promotions on helmets?",
            "I need a composite bat for my son",
            "Do you have leather gloves in stock?",
            "What are the bulk purchase discounts?"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{query}", use_container_width=True):
                st.session_state.example_query = query
    
    # Main chat area
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display agent logs as persistent expander
                if message["role"] == "assistant" and message.get("agent_log"):
                    with st.expander("🤖 Agent Processing Details", expanded=False):
                        for log in message["agent_log"]:
                            # Format log text with HTML line breaks
                            formatted_log = log.replace('\n', '<br>')
                            # Make agent names bold (e.g., "1. Knowledge Agent")
                            import re
                            formatted_log = re.sub(r'(\d+\.)\s+([A-Za-z\s]+Agent)', r'\1 <strong>\2</strong>', formatted_log)
                            st.markdown(f"""
                            <div class="agent-step">
                                {formatted_log}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Display metrics if available
                if message["role"] == "assistant" and "scores" in message and message["scores"]:
                    display_metrics(message["scores"])
    
    # Chat input - always show it
    query = st.chat_input("Ask me about baseball equipment, policies, or promotions...")
    
    # Handle example query from sidebar
    if hasattr(st.session_state, 'example_query'):
        query = st.session_state.example_query
        delattr(st.session_state, 'example_query')
    
    # Process user input
    if query:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Collect agent logs
        agent_log = []
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)
            
            # Show assistant thinking
            with st.chat_message("assistant"):
                # Create status container for agent thoughts with real-time updates
                if show_agent_thoughts:
                    status = st.status("🤔 Processing your request...", expanded=True)
                    with status:
                        st.write("**Multi-Agent System Active**")
                        
                        # Create containers for real-time updates
                        thought_container = st.container()
                        
                        # Custom output handler for real-time streaming
                        class StreamlitOutputHandler:
                            def __init__(self, container):
                                self.container = container
                                self.buffer = []
                                self.current_block = []
                                self.in_chunk_block = False
                                self.in_plan_block = False
                                self.processed_logs = []
                                import threading
                                self.main_thread_id = threading.current_thread().ident
                                
                            def write(self, text):
                                # Skip Streamlit operations if called from worker thread
                                import threading
                                if threading.current_thread().ident != self.main_thread_id:
                                    return  # Silently skip in worker threads
                                
                                # Process each line as it comes
                                lines = text.strip().split('\n')
                                for line in lines:
                                    if line.strip():
                                        self._process_line(line)
                                
                            def _process_line(self, line):
                                # Check if this is the start of a planning block (new format)
                                if "step" in line and "planned" in line:
                                    if self.current_block:
                                        self._display_block()
                                    self.in_plan_block = True
                                    self.in_chunk_block = False
                                    self.current_block.append(line)
                                # Part of plan block - numbered items or quoted subqueries
                                elif self.in_plan_block and (line.strip().startswith(("1.", "2.", "3.", "4.", "5.")) or line.strip().startswith('"')):
                                    self.current_block.append(line)
                                # Plan block ended - when we see a different type of line
                                elif self.in_plan_block and not line.strip().startswith(("1.", "2.", "3.", "4.", "5.", '"')):
                                    self._display_block()
                                    self.in_plan_block = False
                                    if ("Retrieved" in line and "chunks:" in line) or line.strip().startswith("Using subquery:"):
                                        self.in_chunk_block = True
                                        self.current_block.append(line)
                                    else:
                                        self._display_single(line)
                                # Start of chunk block
                                elif ("Retrieved" in line and "chunks:" in line) or line.strip().startswith("Using subquery:"):
                                    if self.current_block:
                                        self._display_block()
                                    self.in_chunk_block = True
                                    self.in_plan_block = False
                                    self.current_block.append(line)
                                # End of chunk block
                                elif self.in_chunk_block and line.strip().startswith("✓ Found"):
                                    self.current_block.append(line)
                                    self._display_block()
                                    self.in_chunk_block = False
                                # Part of current block
                                elif self.in_chunk_block or self.in_plan_block:
                                    self.current_block.append(line)
                                # Regular line
                                else:
                                    if self.current_block:
                                        self._display_block()
                                    self._display_single(line)
                            
                            def _display_block(self):
                                if self.current_block:
                                    block_text = "\n".join(self.current_block)
                                    with self.container:
                                        # Format with HTML line breaks
                                        formatted_text = block_text.replace('\n', '<br>')
                                        # Make agent names bold
                                        import re
                                        formatted_text = re.sub(r'(\d+\.)\s+([A-Za-z\s]+Agent)', r'\1 <strong>\2</strong>', formatted_text)
                                        st.markdown(f"""
                                        <div class="agent-step">
                                            {formatted_text}
                                        </div>
                                        """, unsafe_allow_html=True)
                                    self.processed_logs.append(block_text)
                                    self.current_block = []
                            
                            def _display_single(self, line):
                                with self.container:
                                    # Make agent names bold
                                    import re
                                    formatted_line = re.sub(r'(\d+\.)\s+([A-Za-z\s]+Agent)', r'\1 <strong>\2</strong>', line)
                                    st.markdown(f"""
                                    <div class="agent-step">
                                        {formatted_line}
                                    </div>
                                    """, unsafe_allow_html=True)
                                self.processed_logs.append(line)
                            
                            def flush(self):
                                if self.current_block:
                                    self._display_block()
                        
                        # Use custom handler for real-time streaming
                        handler = StreamlitOutputHandler(thought_container)
                        
                        # Redirect stdout to our handler
                        import sys
                        old_stdout = sys.stdout
                        
                        class TeeOutput:
                            def __init__(self, stream1, stream2):
                                self.stream1 = stream1
                                self.stream2 = stream2
                            
                            def write(self, text):
                                self.stream1.write(text)
                                self.stream2.write(text)
                            
                            def flush(self):
                                self.stream1.flush()
                                if hasattr(self.stream2, 'flush'):
                                    self.stream2.flush()
                        
                        sys.stdout = TeeOutput(old_stdout, handler)
                        
                        try:
                            result = run_mao_with_observability(
                                query=query,
                                session_id=st.session_state.session_id,
                                user_id=st.session_state.user_id,
                                enable_evaluation=enable_evaluation
                            )
                            handler.flush()
                        finally:
                            sys.stdout = old_stdout
                        
                        agent_log = handler.processed_logs
                        
                        status.update(label="✅ Complete!", state="complete", expanded=False)
                else:
                    # Run without showing thoughts
                    with st.spinner("Thinking..."):
                        result = run_mao_with_observability(
                            query=query,
                            session_id=st.session_state.session_id,
                            user_id=st.session_state.user_id,
                            enable_evaluation=False  # Disable during initial response
                        )
                
                # Display response
                answer = result.get("final_answer", "I apologize, but I encountered an error.")
                latency = result.get("latency", 0.0)
                
                # Clean up any formatting issues in the answer
                import re
                answer = answer.replace('\u2028', '\n').replace('\u2029', '\n')  # Line separators
                answer = re.sub(r'\*{3,}', '**', answer)  # Fix triple+ asterisks
                answer = re.sub(r'(\d+)\.(\d+)\s*−', r'$\1.\2 -', answer)  # Fix price formatting
                answer = re.sub(r'•', '-', answer)  # Replace bullet with dash
                
                st.markdown(answer)
                
                # Run evaluation separately if enabled
                scores = {}
                if enable_evaluation:
                    st.markdown("---")
                    eval_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(message: str, current: int, total: int):
                        """Update progress bar and status."""
                        progress = current / total if total > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(message)
                    
                    with st.spinner("Evaluating response quality..."):
                        # Re-run evaluation with progress callback
                        from evaluation import evaluate_response, extract_contexts_from_results
                        
                        contexts = extract_contexts_from_results(
                            result.get("intermediate_results", {})
                        )
                        
                        if contexts:
                            scores = evaluate_response(
                                query=query,
                                answer=answer,
                                contexts=contexts,
                                progress_callback=update_progress
                            )
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                
                # Display evaluation metrics
                if scores:
                    display_metrics(scores)
                
                # Add assistant message to chat with agent log
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "scores": scores,
                    "agent_log": agent_log if show_agent_thoughts else [],
                    "latency": latency
                })
                
                # Show error if any
                if result.get("error"):
                    st.error(f"⚠️ Error: {result['error']}")
        
        # Rerun to show the updated chat and clear the input
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>SOGOSO - Multi-Agent Baseball Equipment Assistant | Powered by LangGraph, LangChain & OpenAI</p>
        <p>Features: Semantic Search • Policy Retrieval • Session Memory • Quality Evaluation</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
