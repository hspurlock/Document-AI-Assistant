import streamlit as st
from pathlib import Path
from agent import AIAgent
from models import ChatSession
import tempfile

st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = AIAgent()
if 'chat_session' not in st.session_state:
    st.session_state.chat_session = ChatSession()
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}  

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return Path(tmp_file.name)

# Sidebar for file upload
with st.sidebar:
    st.title("📚 Document Upload")
    st.write("Supported formats: PDF, DOCX, PPTX, XLSX, TXT, MD, HTML")
    
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=['pdf', 'docx', 'pptx', 'xlsx', 'txt', 'md', 'html'],
        accept_multiple_files=True,
        key='file_uploader'  
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # Save and process file
                    file_path = save_uploaded_file(uploaded_file)
                    
                    # Check if file was already processed
                    is_update = uploaded_file.name in st.session_state.processed_files
                    processed_file = st.session_state.agent.process_file(file_path, is_update=is_update)
                    
                    # Update processed files dict with new checksum
                    if is_update:
                        old_checksum = st.session_state.processed_files[uploaded_file.name]
                        if old_checksum != processed_file.checksum:
                            st.info(f"📝 Updated {uploaded_file.name} with new content")
                        else:
                            st.info(f"ℹ️ {uploaded_file.name} content unchanged")
                    else:
                        st.success(f"✅ {uploaded_file.name} processed successfully!")
                    
                    st.session_state.processed_files[uploaded_file.name] = processed_file.checksum
                    
                    # Clean up temporary file
                    file_path.unlink()
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    
    if st.session_state.processed_files:
        st.write("---")
        st.write("📚 Processed Documents:")
        for filename in st.session_state.processed_files:
            st.write(f"- {filename}")

# Main chat interface
st.title("💬 Chat with your Documents")

# Display chat messages
for message in st.session_state.chat_session.messages:
    with st.chat_message(message.role):
        st.write(message.content)

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Get bot response
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.agent.chat(prompt, st.session_state.chat_session)
            st.write(response)

# Display initial instructions if no messages
if not st.session_state.chat_session.messages:
    st.info("👋 Upload your documents using the sidebar and start asking questions!")
