import streamlit as st
from pathlib import Path
from agent import AIAgent
from models import ChatSession
import tempfile
import os

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
    """Save uploaded file while preserving original filename"""
    try:
        # Create a temporary directory if it doesn't exist
        temp_dir = Path(tempfile.gettempdir()) / "doc_ai_uploads"
        temp_dir.mkdir(exist_ok=True)
        
        # Create file path with original name
        file_path = temp_dir / uploaded_file.name
        
        # Write the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

# Sidebar for document management
with st.sidebar:
    st.title("📚 Document Management")
    
    # Dropdown menu for section selection
    selected_section = st.selectbox(
        "Select Section",
        options=["📤 Upload Documents", "🔍 Browse Documents"],
        key="section_selector"
    )
    
    st.write("---")
    
    # Document Upload Section
    if selected_section == "📤 Upload Documents":
        st.subheader("📤 Document Upload")
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
                        file_path = save_uploaded_file(uploaded_file)
                        if not file_path:
                            continue
                        
                        is_update = uploaded_file.name in st.session_state.processed_files
                        processed_file = st.session_state.agent.process_file(
                            file_path, 
                            is_update=is_update,
                            original_filename=uploaded_file.name
                        )
                        
                        if not processed_file:
                            st.error(f"❌ Failed to process {uploaded_file.name}")
                            if file_path.exists():
                                file_path.unlink()
                            continue
                        
                        if is_update:
                            old_checksum = st.session_state.processed_files[uploaded_file.name]
                            if old_checksum != processed_file.checksum:
                                st.info(f"📝 Updated {uploaded_file.name}")
                                st.session_state.processed_files[uploaded_file.name] = processed_file.checksum
                            else:
                                st.info(f"ℹ️ {uploaded_file.name} unchanged")
                        else:
                            st.success(f"✅ {uploaded_file.name}")
                            st.session_state.processed_files[uploaded_file.name] = processed_file.checksum
                        
                        if file_path.exists():
                            file_path.unlink()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        if file_path and file_path.exists():
                            file_path.unlink()
        
        if st.session_state.processed_files:
            st.write("---")
            st.write("📚 Processed Documents:")
            for filename, checksum in st.session_state.processed_files.items():
                st.write(f"- {filename}")
    
    # Vector Store Browser Section
    if selected_section == "🔍 Browse Documents":
        st.subheader("🔍 Vector Store Browser")
        
        # Get vector store contents
        contents = st.session_state.agent.get_vector_store_contents()
        
        if not contents['documents']:
            st.info("No documents found. Upload some documents first!")
        else:
            # Document selection
            doc_options = []
            for source, doc in contents['documents'].items():
                if doc['chunks']:
                    filename = doc['chunks'][0].get('filename', Path(source).name)
                    doc_options.append((source, filename))
            
            doc_options.sort(key=lambda x: x[1].lower())
            
            selected_doc = st.selectbox(
                "📑 Select Document",
                options=[source for source, _ in doc_options],
                format_func=lambda x: next(filename for s, filename in doc_options if s == x)
            )
            
            doc = contents['documents'][selected_doc]
            filename = next(filename for s, filename in doc_options if s == selected_doc)
            
            # Document info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pages", len(doc['pages']))
            with col2:
                st.metric("Chunks", doc['total_chunks'])
            with col3:
                if st.button("🗑️ Delete", type="secondary", use_container_width=True):
                    if st.session_state.agent.delete_document(selected_doc):
                        st.success(f"Deleted {filename}")
                        st.rerun()
                    else:
                        st.error(f"Failed to delete {filename}")
            
            # Content view
            show_content = st.toggle("Show content", value=False)
            
            if show_content:
                # Group chunks by page
                pages = {}
                for chunk in doc['chunks']:
                    page = chunk['page']
                    if page not in pages:
                        pages[page] = []
                    pages[page].append(chunk)
                
                # Display pages
                for page_num in sorted(pages.keys()):
                    page_chunks = pages[page_num]
                    
                    with st.expander(f"📄 Page {page_num}", expanded=False):
                        # Show first chunk
                        if page_chunks:
                            st.markdown(f"```text\n{page_chunks[0]['text']}\n```")
                            
                            if len(page_chunks) > 1:
                                st.markdown(f"```text\n{page_chunks[-1]['text']}\n```")
                            
                            if len(page_chunks) > 2:
                                show_all = st.toggle('Show all', key=f"toggle_{page_num}")
                                if show_all:
                                    for chunk in page_chunks[1:-1]:
                                        st.markdown(f"```text\n{chunk['text']}\n```")
            
            # Search
            st.write("---")
            search_term = st.text_input("🔎 Search", placeholder="Enter search term...")
            
            if search_term:
                for chunk in doc['chunks']:
                    if search_term.lower() in chunk['text'].lower():
                        with st.expander(f"📄 Page {chunk['page']} - Match", expanded=True):
                            text = chunk['text'].replace(search_term, f"**{search_term}**")
                            st.markdown(text)

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
