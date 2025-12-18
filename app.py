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
    
    /* Intelligence Trace Styles */
    .trace-container {
        background: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 6px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        margin-bottom: 1rem;
        overflow: hidden;
    }
    
    .trace-header {
        background: #f6f8fa;
        padding: 8px 16px;
        border-bottom: 1px solid #e1e4e8;
        font-size: 0.85rem;
        color: #24292e;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .trace-header-status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
    }
    
    .trace-header-meta {
        color: #586069;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 0.75rem;
    }
    
    .trace-content {
        padding: 0;
    }
    
    .trace-phase {
        padding: 12px 16px;
        border-bottom: 1px solid #eaecef;
        display: flex;
        gap: 16px;
    }
    
    .trace-phase:last-child {
        border-bottom: none;
    }
    
    .phase-icon {
        width: 24px;
        display: flex;
        justify-content: center;
        padding-top: 2px;
        font-size: 1.2rem;
    }
    
    .phase-body {
        flex: 1;
    }
    
    .phase-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #24292e;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    
    .trace-badge {
        display: inline-flex;
        align-items: center;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        border: 1px solid transparent;
    }
    
    .badge-intent {
        background-color: #dbedff;
        color: #0366d6;
        border-color: #c8e1ff;
    }
    
    .badge-entity {
        background-color: #f1f8ff;
        color: #24292e;
        border-color: #e1e4e8;
    }
    
    .badge-label {
        font-weight: 600;
        margin-right: 4px;
        opacity: 0.7;
        font-size: 0.7rem;
        text-transform: uppercase;
    }
    
    .checklist-item {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 6px;
        font-size: 0.9rem;
        color: #24292e;
    }
    
    .checklist-check {
        color: #22863a;
        font-weight: bold;
    }
    
    .subquery-code {
        display: block;
        margin-top: 4px;
        padding: 4px 8px;
        background-color: #f6f8fa;
        border-radius: 4px;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 0.8rem;
        color: #444;
        border: 1px solid #eaecef;
    }
    
    .agent-card {
        background: #fff;
        border: 1px solid #e1e4e8;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .agent-info {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .agent-name {
        font-weight: 600;
        font-size: 0.9rem;
        color: #24292e;
    }
    
    .agent-detail {
        font-size: 0.8rem;
        color: #586069;
    }
    
    .agent-meta {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 0.75rem;
        color: #6a737d;
    }
    
    .success-icon {
        color: #22863a;
    }
    
    /* Quality Footer Styles */
    .quality-footer {
        margin-top: 12px;
        padding-top: 8px;
        border-top: 1px solid #f0f0f0;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
    }
    
    .quality-pill {
        display: inline-flex;
        align-items: center;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid transparent;
    }
    
    .pill-high {
        background-color: #dafbe1;
        color: #1a7f37;
        border-color: #ccebd4;
    }
    
    .pill-medium {
        background-color: #fff8c5;
        color: #9a6700;
        border-color: #fbeeb8;
    }
    
    .pill-low {
        background-color: #ffebe9;
        color: #cf222e;
        border-color: #ffcecb;
    }
    
    .quality-label {
        margin-right: 4px;
        opacity: 0.8;
    }
    
    .quality-value {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }
</style>
""", unsafe_allow_html=True)


def render_quality_footer(scores):
    """
    Render the quality evaluation scores as a sleek footer.
    
    Args:
        scores: Dictionary of evaluation scores
    """
    if not scores:
        return
        
    html = '<div class="quality-footer">'
    
    for metric, score in scores.items():
        # Determine color class
        if score >= 0.8:
            color_class = "pill-high"
        elif score >= 0.5:
            color_class = "pill-medium"
        else:
            color_class = "pill-low"
            
        # Format score
        score_fmt = f"{score:.2f}"
        
        html += f"""
        <div class="quality-pill {color_class}">
            <span class="quality-label">{metric}:</span>
            <span class="quality-value">{score_fmt}</span>
        </div>
        """
        
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_intelligence_trace(logs):
    """
    Render the agent logs as a beautiful Intelligence Trace UI.
    
    Args:
        logs: List of log strings from the agent execution
    """
    if not logs:
        return
        
    # Parse logs into structured data
    phases = {
        "understanding": {"title": "Understanding", "icon": "🔍", "items": []},
        "strategy": {"title": "Strategy", "icon": "🧠", "items": []},
        "execution": {"title": "Execution", "icon": "⚡", "items": []}
    }
    
    current_phase = "understanding"
    
    # Helper to extract value from log line
    def extract_val(line, prefix):
        if "|" in line:
            return line.split("|")[1].strip()
        return line.replace(prefix, "").strip()
    
    # Parse the logs
    for log in logs:
        log = log.strip()
        
        # Phase detection
        if "Analyzing User Intent" in log:
            current_phase = "understanding"
        elif "Planning Execution" in log:
            current_phase = "strategy"
        elif "Searching" in log or "Checking" in log or "Generating" in log:
            current_phase = "execution"
            
        # Content extraction
        if log.startswith("GLASS:"):
            if current_phase == "understanding":
                if "Intent" in log:
                    phases["understanding"]["items"].append({"type": "intent", "value": extract_val(log, "GLASS: Intent")})
                elif "Product" in log:
                    phases["understanding"]["items"].append({"type": "entity", "label": "PRODUCT", "value": extract_val(log, "GLASS: Product")})
                elif "Brand" in log:
                    phases["understanding"]["items"].append({"type": "entity", "label": "BRAND", "value": extract_val(log, "GLASS: Brand")})
                elif "Price" in log:
                    phases["understanding"]["items"].append({"type": "entity", "label": "PRICE", "value": extract_val(log, "GLASS: Max Price")})
                    
            elif current_phase == "strategy":
                if "Plan" in log:
                    # Parse plan list
                    plan_str = extract_val(log, "GLASS: Plan")
                    agents = [a.strip() for a in plan_str.split(",")]
                    for agent in agents:
                        phases["strategy"]["items"].append({"type": "plan_item", "agent": agent, "subquery": None})
                elif "Subquery" in log:
                    # Attach subquery to last plan item or create new
                    subquery = extract_val(log, "GLASS: Subquery")
                    # Find which agent this belongs to
                    agent_name = log.split("(")[1].split(")")[0] if "(" in log else "unknown"
                    
                    # Update existing item if found
                    found = False
                    for item in phases["strategy"]["items"]:
                        if item["type"] == "plan_item" and item["agent"] == agent_name:
                            item["subquery"] = subquery
                            found = True
                            break
                    if not found:
                        phases["strategy"]["items"].append({"type": "plan_item", "agent": agent_name, "subquery": subquery})
                        
            elif current_phase == "execution":
                if "Query" in log:
                    phases["execution"]["items"].append({"type": "query", "value": extract_val(log, "GLASS: Query")})
                    
        elif log.startswith("INFO:"):
            if current_phase == "execution":
                if "Found" in log:
                    phases["execution"]["items"].append({"type": "result", "value": log.replace("INFO:", "").strip()})
                elif "Generated" in log:
                    phases["execution"]["items"].append({"type": "result", "value": "Response synthesized successfully"})
                    
        elif log.startswith("STEP:"):
            if current_phase == "execution":
                # Add agent execution step
                step_title = log.split("|")[0].replace("STEP:", "").strip()
                phases["execution"]["items"].append({"type": "step", "title": step_title})

    # Generate HTML
    html_parts = []
    html_parts.append(f"""
    <div class="trace-container">
        <div class="trace-header">
            <div class="trace-header-status">
                <span class="success-icon">✓</span> Request Processed
            </div>
            <div class="trace-header-meta">
                {len(phases['strategy']['items'])} Agents Active
            </div>
        </div>
        <div class="trace-content">
    """)
    
    # Phase 1: Understanding
    if phases["understanding"]["items"]:
        html_parts.append(f"""
        <div class="trace-phase">
            <div class="phase-icon">{phases['understanding']['icon']}</div>
            <div class="phase-body">
                <div class="phase-title">{phases['understanding']['title']}</div>
                <div class="badge-container">
        """)
        for item in phases["understanding"]["items"]:
            if item["type"] == "intent":
                html_parts.append(f"""
                <div class="trace-badge badge-intent">
                    <span class="badge-label">INTENT</span> {item['value']}
                </div>
                """)
            else:
                html_parts.append(f"""
                <div class="trace-badge badge-entity">
                    <span class="badge-label">{item['label']}</span> {item['value']}
                </div>
                """)
        html_parts.append("""
                </div>
            </div>
        </div>
        """)
        
    # Phase 2: Strategy
    if phases["strategy"]["items"]:
        html_parts.append(f"""
        <div class="trace-phase">
            <div class="phase-icon">{phases['strategy']['icon']}</div>
            <div class="phase-body">
                <div class="phase-title">{phases['strategy']['title']}</div>
        """)
        for item in phases["strategy"]["items"]:
            agent_display = item['agent'].replace('_agent', '').title() + " Agent"
            html_parts.append(f"""
            <div class="checklist-item">
                <span class="checklist-check">✓</span>
                <div style="flex: 1">
                    <div>Activate <strong>{agent_display}</strong></div>
                    {f'<div class="subquery-code">{item["subquery"]}</div>' if item["subquery"] else ''}
                </div>
            </div>
            """)
        html_parts.append("""
            </div>
        </div>
        """)

    # Phase 3: Execution
    if phases["execution"]["items"]:
        html_parts.append(f"""
        <div class="trace-phase">
            <div class="phase-icon">{phases['execution']['icon']}</div>
            <div class="phase-body">
                <div class="phase-title">{phases['execution']['title']}</div>
        """)
        
        # Group execution items by step
        current_step = None
        for item in phases["execution"]["items"]:
            if item["type"] == "step":
                # Close previous card if exists
                if current_step:
                    html_parts.append("</div></div>")
                
                # Start new card
                title = item["title"]
                html_parts.append(f"""
                <div class="agent-card">
                    <div class="agent-info">
                        <span class="success-icon">✓</span>
                        <div>
                            <div class="agent-name">{title}</div>
                """)
                current_step = True
            elif item["type"] == "result" and current_step:
                html_parts.append(f"""
                            <div class="agent-detail">{item['value']}</div>
                """)
            elif item["type"] == "query" and current_step:
                 pass # Skip showing raw query in execution card to keep it clean
                 
        if current_step:
            html_parts.append("</div></div><div class='agent-meta'>0.4s</div></div>")
            
        html_parts.append("""
            </div>
        </div>
        """)

    html_parts.append("""
        </div>
    </div>
    """)
    
    # Join and clean up indentation to prevent code block rendering
    full_html = "\n".join(html_parts)
    import textwrap
    # Simple way to strip indentation from each line
    clean_html = "\n".join([line.strip() for line in full_html.split("\n")])
    
    st.markdown(clean_html, unsafe_allow_html=True)


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
        "Context Recall": "📈",
        "Professionalism": "👔",
        "Context Utilization": "📊"
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
        
        # Session Quality Metrics
        if st.session_state.messages:
            # Calculate average quality score across all assistant messages
            total_score = 0
            count = 0
            for msg in st.session_state.messages:
                if msg["role"] == "assistant" and msg.get("scores"):
                    # Average of all metrics for this message
                    msg_scores = msg["scores"].values()
                    if msg_scores:
                        msg_avg = sum(msg_scores) / len(msg_scores)
                        total_score += msg_avg
                        count += 1
            
            if count > 0:
                session_avg = total_score / count
                st.markdown("### Session Quality Metrics")
                st.markdown(f"**Average Quality Score:** {session_avg:.2f}/1.0")
                st.progress(session_avg)
                st.caption(f"Based on {count} evaluated queries")
        
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
            
        # Debug Mode
        with st.expander("🐞 Debug State"):
            st.write("Session State:", st.session_state)
            if "messages" in st.session_state and st.session_state.messages:
                st.write("Last Message:", st.session_state.messages[-1])
                if "scores" in st.session_state.messages[-1]:
                    st.write("Last Message Scores:", st.session_state.messages[-1]["scores"])
            
            # Check for graph state if available (it's not directly accessible here but we can show what we have)
            if "last_result" in st.session_state:
                 st.write("Last Graph Result:", st.session_state.last_result)
        
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
                    with st.expander("🤖 Intelligence Trace", expanded=False):
                        render_intelligence_trace(message["agent_log"])
                
                # Display quality footer if scores exist
                if message["role"] == "assistant" and "scores" in message and message["scores"]:
                    render_quality_footer(message["scores"])
    
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
                            def __init__(self, container, status_container):
                                self.container = container
                                self.status_container = status_container
                                self.buffer = []
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
                                line = line.strip()
                                if line.startswith("STEP:"):
                                    # Format: STEP: Title | Subtitle
                                    parts = line[5:].split("|")
                                    title = parts[0].strip()
                                    subtitle = parts[1].strip() if len(parts) > 1 else ""
                                    
                                    # Update status label
                                    self.status_container.update(label=title, state="running")
                                    
                                    # Show subtitle in container
                                    if subtitle:
                                        with self.container:
                                            st.markdown(f"**{subtitle}**")
                                    
                                    self.processed_logs.append(line)
                                    
                                elif line.startswith("GLASS:"):
                                    # Format: GLASS: Label | Value
                                    parts = line[6:].split("|")
                                    label = parts[0].strip()
                                    value = parts[1].strip() if len(parts) > 1 else ""
                                    
                                    with self.container:
                                        st.markdown(f"""
                                        <div style="display: flex; align-items: center; margin: 5px 0;">
                                            <span style="font-weight: bold; margin-right: 10px;">{label}:</span>
                                            <span style="background-color: #e6f3ff; padding: 2px 8px; border-radius: 4px; color: #0066cc;">{value}</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    self.processed_logs.append(line)
                                    
                                elif line.startswith("INFO:"):
                                    # Format: INFO: Message
                                    msg = line[5:].strip()
                                    with self.container:
                                        st.markdown(f"<div style='color: #666; font-size: 0.9em; margin-left: 10px;'>• {msg}</div>", unsafe_allow_html=True)
                                    self.processed_logs.append(line)
                                    
                                else:
                                    # Fallback for other logs (legacy support)
                                    if not any(line.startswith(p) for p in ["STEP:", "GLASS:", "INFO:"]):
                                        # Only display if it looks like a meaningful log
                                        if "Agent" in line or "Retrieved" in line or "Found" in line:
                                            with self.container:
                                                st.markdown(f"<div style='color: #888; font-size: 0.8em;'>{line}</div>", unsafe_allow_html=True)
                                            self.processed_logs.append(line)
                            
                            def flush(self):
                                pass
                        
                        # Use custom handler for real-time streaming
                        handler = StreamlitOutputHandler(thought_container, status)
                        
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
                            enable_evaluation=enable_evaluation
                        )
                
                # Store result for debug
                st.session_state.last_result = result

                # Display response
                answer = result.get("final_answer", "I apologize, but I encountered an error.")
                latency = result.get("latency", 0.0)
                
                # Clean up any formatting issues in the answer
                import re
                # Remove line separators that cause vertical text issues instead of replacing with newlines
                # Using space to prevent words from merging if separator was used as space
                answer = answer.replace('\u2028', ' ').replace('\u2029', ' ')
                
                # Fix missing spaces after punctuation (e.g. "200.However" -> "200. However")
                answer = re.sub(r'([a-z0-9]\.)([A-Z])', r'\1 \2', answer)
                
                answer = re.sub(r'\*{3,}', '**', answer)  # Fix triple+ asterisks
                answer = re.sub(r'(\d+)\.(\d+)\s*−', r'$\1.\2 -', answer)  # Fix price formatting
                answer = re.sub(r'•', '-', answer)  # Replace bullet with dash
                
                st.markdown(answer)
                
                # Get scores from result or run evaluation if missing
                scores = result.get("evaluation_scores", {})
                
                # Fallback: Run evaluation separately if enabled but not returned by backend
                if enable_evaluation and not scores:
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
                            try:
                                scores = evaluate_response(
                                    query=query,
                                    answer=answer,
                                    contexts=contexts,
                                    progress_callback=update_progress
                                )
                            except Exception as e:
                                st.warning(f"Evaluation failed: {e}")
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                
                # Display evaluation metrics
                if scores:
                    render_quality_footer(scores)
                
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
