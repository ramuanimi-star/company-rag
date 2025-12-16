import os
import warnings
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['POSTHOG_DISABLED'] = 'True'
warnings.filterwarnings('ignore')

import streamlit as st
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

st.set_page_config(
    page_title="CGI HR Assistant",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with CGI branding
st.markdown("""
<style>
    /* CGI Red color scheme */
    :root {
        --cgi-red: #d32f2f;
        --cgi-dark-red: #b71c1c;
        --cgi-light-red: #ef5350;
    }
    
    /* Main title styling with CGI branding */
    .cgi-header {
        background: linear-gradient(135deg, #d32f2f 0%, #b71c1c 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(211, 47, 47, 0.3);
    }
    
    .cgi-logo-text {
        font-size: 3rem;
        font-weight: 900;
        color: white;
        letter-spacing: 0.3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .cgi-tagline {
        color: #ffcdd2;
        font-size: 1.1rem;
        font-weight: 300;
        letter-spacing: 0.1rem;
        margin-bottom: 0.5rem;
    }
    
    .cgi-subtitle {
        color: white;
        font-size: 1.3rem;
        font-weight: 500;
        margin-top: 1rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffebee 0%, #ffcdd2 100%);
    }
    
    /* Metric cards with CGI red */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #d32f2f;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #fafafa;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #d32f2f;
    }
    
    /* Buttons with CGI red */
    .stButton>button {
        background-color: #d32f2f;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #b71c1c;
        box-shadow: 0 4px 8px rgba(211, 47, 47, 0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #ffebee;
        border-radius: 8px;
        font-weight: 500;
        border-left: 3px solid #d32f2f;
    }
    
    /* Document list */
    .doc-item {
        background: white;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid #d32f2f;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s;
    }
    
    .doc-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(211, 47, 47, 0.2);
    }
    
    /* Example questions */
    .example-q {
        background: white;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 8px;
        border-left: 3px solid #d32f2f;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .example-q:hover {
        background: #ffebee;
        transform: translateX(5px);
    }
    
    /* Mode badge */
    .mode-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .rag-mode {
        background-color: #e3f2fd;
        color: #1976d2;
        border: 1px solid #1976d2;
    }
    
    .llm-mode {
        background-color: #fff3e0;
        color: #f57c00;
        border: 1px solid #f57c00;
    }
    
    /* Info boxes */
    .info-box {
        background: #e8f5e9;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .sidebar-header {
        color: #d32f2f;
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_rag():
    try:
        # Initialize LLM with faster model
        Settings.llm = Ollama(
            model="llama3.2:3b",  # Much faster!
            request_timeout=120.0,  # Reduced timeout
            additional_kwargs={
                "num_predict": 256,  # Shorter responses
                "temperature": 0.7,
                "num_ctx": 2048  # Smaller context
            }
        )
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
        
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = chroma_client.get_or_create_collection("hr_documents")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
        # Optimized query engine
        query_engine = index.as_query_engine(
            similarity_top_k=2,
            response_mode="compact"
        )
        
        return query_engine, chroma_collection.count(), Settings.llm, None
    except Exception as e:
        return None, 0, None, str(e)

def is_document_related(question):
    """Improved detection for document vs generic questions"""
    
    question_lower = question.lower().strip()
    
    # Strong generic indicators - use LLM
    generic_patterns = [
        'hello', 'hi ', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'what can you do', 'who are you', 'help',
        'can you help', 'can you act', 'act as', 'introduce yourself',
        'what are you', 'tell me about yourself', 'your name', 'your purpose',
        'thank you', 'thanks', 'bye', 'goodbye', 'what is cgi', 'about cgi'
    ]
    
    for pattern in generic_patterns:
        if pattern in question_lower:
            return False
    
    # Strong document indicators - use RAG
    doc_keywords = [
        'transition cost', 'contract cost', 'time report', 'timesheet',
        'according to', 'in the document', 'in the policy', 'page ',
        'what does the document', 'find in', 'search for', 'procedure for',
        'how to submit', 'approval process', 'guideline for', 'form for'
    ]
    
    for keyword in doc_keywords:
        if keyword in question_lower:
            return True
    
    # Default: short questions use LLM, longer ones use RAG
    return len(question_lower.split()) > 8

def get_llm_response(llm, question):
    """Get direct response from LLM"""
    response = llm.complete(
        f"You are a helpful HR assistant at CGI. Answer concisely in 2-3 sentences.\n\nQuestion: {question}\n\nAnswer:"
    )
    return str(response)

# Initialize system
query_engine, doc_count, llm, error = init_rag()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode_preference" not in st.session_state:
    st.session_state.mode_preference = "auto"

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">🔴 CGI HR ASSISTANT</div>', unsafe_allow_html=True)
    st.markdown("*Insights you can act on*")
    st.markdown("---")
    
    # System status
    if error:
        st.error("❌ **System Offline**")
        st.caption(f"Error: {error}")
        st.info("💡 Run `python rag_app.py` first")
    else:
        st.success("✅ **System Online**")
        
        col1, col2 = st.columns(2)
        with col1:
            num_docs = len([f for f in os.listdir("./data") if f.endswith('.pdf')]) if os.path.exists("./data") else 0
            st.metric("📄 Docs", num_docs)
        with col2:
            st.metric("📦 Chunks", doc_count)
    
    st.markdown("---")
    
    # Response mode selection
    st.markdown('<div class="sidebar-header">🎯 Response Mode</div>', unsafe_allow_html=True)
    
    mode_option = st.radio(
        "Choose how I should respond:",
        ["🤖 Auto (Smart)", "💬 Chat Only (Fast)", "📚 Documents Only"],
        help="""
        **Auto**: Automatically detects question type
        **Chat Only**: Quick responses without document search
        **Documents Only**: Always searches your documents
        """
    )
    
    if mode_option == "🤖 Auto (Smart)":
        st.session_state.mode_preference = "auto"
        st.markdown('<div class="info-box">✨ Smart mode will choose the best response method automatically</div>', unsafe_allow_html=True)
    elif mode_option == "💬 Chat Only (Fast)":
        st.session_state.mode_preference = "llm"
        st.markdown('<div class="info-box">⚡ Fast mode - perfect for greetings and general questions</div>', unsafe_allow_html=True)
    else:
        st.session_state.mode_preference = "rag"
        st.markdown('<div class="info-box">📚 Document search mode - searches all indexed files</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Documents
    st.markdown('<div class="sidebar-header">📚 Indexed Documents</div>', unsafe_allow_html=True)
    if os.path.exists("./data"):
        pdf_files = [f for f in os.listdir("./data") if f.lower().endswith('.pdf')]
        if pdf_files:
            for pdf in pdf_files:
                display_name = pdf if len(pdf) <= 35 else pdf[:32] + "..."
                st.markdown(f'<div class="doc-item">📄 {display_name}</div>', unsafe_allow_html=True)
        else:
            st.info("No documents indexed yet")
    
    st.markdown("---")
    
    # Examples
    st.markdown('<div class="sidebar-header">💡 Try These Questions</div>', unsafe_allow_html=True)
    
    st.markdown("**💬 General (Chat Mode):**")
    st.markdown('<div class="example-q">• Good morning!</div>', unsafe_allow_html=True)
    st.markdown('<div class="example-q">• What is CGI?</div>', unsafe_allow_html=True)
    
    st.markdown("**📚 Specific (Document Mode):**")
    st.markdown('<div class="example-q">• What are transition costs?</div>', unsafe_allow_html=True)
    st.markdown('<div class="example-q">• How do I submit time reports?</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System info
    st.markdown('<div class="sidebar-header">⚙️ System Info</div>', unsafe_allow_html=True)
    st.caption("🤖 Model: Qwen2.5 7B")
    st.caption("🔒 Privacy: 100% Local")
    st.caption("⚡ Status: Optimized")
    st.caption("🏢 CGI Inc.")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main area
st.markdown("""
<div class="cgi-header">
    <div class="cgi-logo-text">CGI</div>
    <div class="cgi-tagline">INSIGHTS YOU CAN ACT ON</div>
    <div class="cgi-subtitle">🤖 HR Knowledge Assistant</div>
</div>
""", unsafe_allow_html=True)

# Mode indicator
if st.session_state.mode_preference == "auto":
    mode_text = "🤖 **Auto Mode** - I'll choose the best way to answer your questions"
elif st.session_state.mode_preference == "llm":
    mode_text = "💬 **Chat Mode** - Fast responses without document search"
else:
    mode_text = "📚 **Document Mode** - Searching your indexed files"

st.markdown(f'<p style="color: #666; font-size: 1rem; margin-bottom: 2rem;">{mode_text}</p>', unsafe_allow_html=True)

if not query_engine:
    st.error("⚠️ System not initialized. Run `python rag_app.py` first.")
    st.stop()

# Welcome message
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant", avatar="🔴"):
        st.markdown("""
        👋 **Welcome to CGI HR Assistant!**
        
        I can help you in two ways:
        
        💬 **Chat Mode** - Quick answers to general questions (fast!)  
        📚 **Document Mode** - Search specific information in your HR documents  
        
        **Current mode:** """ + ("🤖 Auto (Smart)" if st.session_state.mode_preference == "auto" 
                                else "💬 Chat Only" if st.session_state.mode_preference == "llm" 
                                else "📚 Documents Only") + """
        
        Change the mode anytime in the sidebar! Ask me anything 👇
        """)

# Display chat history
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🔴"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant" and "mode" in msg:
            if msg["mode"] == "RAG":
                st.markdown('<span class="mode-badge rag-mode">📚 DOCUMENT SEARCH</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="mode-badge llm-mode">💬 CHAT MODE</span>', unsafe_allow_html=True)
        
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("📚 **Source Documents**"):
                for i, source in enumerate(msg["sources"], 1):
                    st.markdown(f"**{i}.** 📄 `{source['file']}` - Page **{source['page']}** ({source['score']:.1%})")

# Chat input
if prompt := st.chat_input("💬 Type your question here...", key="chat_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="🔴"):
        # Determine mode
        if st.session_state.mode_preference == "auto":
            use_rag = is_document_related(prompt)
        elif st.session_state.mode_preference == "llm":
            use_rag = False
        else:
            use_rag = True
        
        mode = "RAG" if use_rag else "LLM"
        
        # Show badge
        if mode == "RAG":
            st.markdown('<span class="mode-badge rag-mode">📚 DOCUMENT SEARCH</span>', unsafe_allow_html=True)
            spinner_text = "🔍 Searching documents..."
        else:
            st.markdown('<span class="mode-badge llm-mode">💬 CHAT MODE</span>', unsafe_allow_html=True)
            spinner_text = "💭 Thinking..."
        
        with st.spinner(spinner_text):
            try:
                if use_rag:
                    # Document search mode
                    response = query_engine.query(prompt)
                    answer = str(response)
                    
                    sources = []
                    if hasattr(response, 'source_nodes') and response.source_nodes:
                        for node in response.source_nodes:
                            sources.append({
                                "file": node.node.metadata.get('file_name', 'Unknown'),
                                "page": node.node.metadata.get('page_label', 'N/A'),
                                "score": node.score if hasattr(node, 'score') else 0
                            })
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📚 **Source Documents**"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**{i}.** 📄 `{source['file']}` - Page **{source['page']}** ({source['score']:.1%})")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "mode": "RAG"
                    })
                    
                else:
                    # Chat mode
                    answer = get_llm_response(llm, prompt)
                    st.markdown(answer)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": [],
                        "mode": "LLM"
                    })
                
            except Exception as e:
                error_msg = f"❌ **Error:** {str(e)}"
                st.error(error_msg)
                
                if "timed out" in str(e).lower():
                    st.warning("⏱️ Timeout. Try Chat Mode for faster responses or ask a shorter question.")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "mode": mode
                })

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #999; font-size: 0.85rem; padding: 1rem;">'
    '🔴 <strong>CGI Inc.</strong> - Member of CGI Group<br>'
    '🔒 100% Local Processing • No Data Leaves Your Computer<br>'
    'Powered by Qwen2.5 & LlamaIndex<br>'
    '<em>Insights you can act on</em>'
    '</div>',
    unsafe_allow_html=True
)