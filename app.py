from pathlib import Path
import tempfile
import streamlit as st
from time import time
from agent import AIAgent
from models import ChatSession, ChatMessage
from security import FileValidator, SessionManager, sanitize_input

# Initialize session state
if 'upload_status_messages' not in st.session_state:
    st.session_state.upload_status_messages = []

if 'last_activity' not in st.session_state:
    st.session_state.last_activity = time()

# Check session timeout
if SessionManager.check_session_timeout(st.session_state.last_activity):
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Update last activity time
st.session_state.last_activity = time()

if 'agent' not in st.session_state:
    models = AIAgent.get_available_models()
    st.session_state.selected_model = models[0] if models else "-"
    st.session_state.agent = AIAgent(model_name=st.session_state.selected_model)

if 'selected_docs' not in st.session_state:
    st.session_state.selected_docs = set()

if 'doc_cache_key' not in st.session_state:
    st.session_state.doc_cache_key = 0

if 'checkbox_key' not in st.session_state:
    st.session_state.checkbox_key = 0

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = ChatSession()

if 'selected_model' not in st.session_state:
    models = st.session_state.agent.get_available_models()
    st.session_state.selected_model = models[0] if models else "-"

if 'last_prompt' not in st.session_state:
    st.session_state.last_prompt = None

if 'form_submit_key' not in st.session_state:
    st.session_state.form_submit_key = 0

# Configure page
st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling - Using safer CSS
st.markdown("""
<style>
    /* Safe styling using data attributes */
    [data-testid="MainMenu"] {visibility: hidden;}
    [data-testid="stFooter"] {visibility: hidden;}
    
    /* Button styling */
    [data-testid="stButton"] button {width: 100%;}
    
    /* Hide file list - using safer selectors */
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] {display: none;}
    
    /* Chat interface styling - using safer selectors */
    [data-testid="stChatMessage"] {max-width: 800px; margin: 1rem auto;}
    
    /* Container styling */
    .main .block-container {padding-bottom: 100px;}
    
    /* Chat input styling */
    .stChatInputContainer {background: var(--background-color); padding: 1rem;}
</style>
""", unsafe_allow_html=True)

# Get current documents from vector store
def get_vector_store_contents():
    return st.session_state.agent.get_vector_store_contents()

# Get fresh document data using cache key
vector_store_contents = get_vector_store_contents()
documents = vector_store_contents

# Sidebar
with st.sidebar:
    st.title("Document AI Assistant")
    
    # Model selector
    st.subheader("🤖 Model")
    models = st.session_state.agent.get_available_models()
    if not models:
        st.warning("No models available. Please make sure Ollama is running.")
        st.selectbox("Select Model", ["-"], label_visibility="collapsed", disabled=True)
    else:
        selected = st.selectbox(
            "Select Model",
            options=models,
            index=models.index(st.session_state.selected_model) if st.session_state.selected_model in models else 0,
            label_visibility="collapsed"
        )
        if selected != st.session_state.selected_model:
            st.session_state.selected_model = selected
            # Reinitialize agent with new model
            st.session_state.agent = AIAgent(model_name=selected)
            st.rerun()
    
    st.write("---")
    
    # Upload documents and images
    st.subheader("📤 Upload Files")
    st.write("📄 Documents: PDF, DOCX, TXT, MD")
    st.write("🖼️ Images: PNG, JPG, JPEG, GIF, BMP")
    
    # Display any persistent status messages
    for msg in st.session_state.upload_status_messages:
        if msg['type'] == 'error':
            st.error(msg['text'], icon='❌')
        elif msg['type'] == 'warning':
            st.warning(msg['text'], icon='⚠️')
        elif msg['type'] == 'info':
            st.info(msg['text'], icon='ℹ️')
    
    # Add clear messages button if there are messages
    if st.session_state.upload_status_messages:
        if st.button('Clear Messages', key='clear_status_messages'):
            st.session_state.upload_status_messages = []
            st.rerun()
    
    status_container = st.empty()
    status_col1, status_col2 = st.columns([20, 1])
    with status_col1:
        status_message = st.empty()
    with status_col2:
        close_button = st.empty()
    
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["txt", "md", "docx", "pdf", "png", "jpg", "jpeg", "gif", "bmp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"doc_uploader_{st.session_state.doc_cache_key}"
    )

    has_errors = False
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Validate file
            is_valid, error_msg = FileValidator.validate_file(uploaded_file.name, uploaded_file.size)
            if not is_valid:
                st.session_state.upload_status_messages.append({
                    'type': 'error',
                    'text': f"Invalid file {uploaded_file.name}: {error_msg}"
                })
                has_errors = True
                continue
                
            # Check if file was already processed in this session
            file_key = f"processed_{uploaded_file.name}_{st.session_state.doc_cache_key}"
            if file_key not in st.session_state:
                st.session_state[file_key] = False
            
            if not st.session_state[file_key]:
                # Get file extension
                ext = uploaded_file.name.rsplit('.', 1)[1].lower() if '.' in uploaded_file.name else ''
                is_image = ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp']
                
                # Show appropriate processing message
                if is_image:
                    status_message.info(f"🔍 Processing image: {uploaded_file.name}...")
                else:
                    status_message.info(f"📄 Processing document: {uploaded_file.name}...")
                
                try:
                    file_path = Path(tempfile.gettempdir()) / "doc_ai_uploads" / uploaded_file.name
                    file_path.parent.mkdir(exist_ok=True)
                    
                    # Write the file
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    is_update = uploaded_file.name in documents.get('documents', {})
                    processed_file = st.session_state.agent.process_file(
                        file_path, 
                        is_update=is_update,
                        original_filename=uploaded_file.name
                    )
                    
                    if not processed_file:
                        st.session_state.upload_status_messages.append({
                            'type': 'error',
                            'text': f"Failed to process {uploaded_file.name}"
                        })
                        if file_path.exists():
                            file_path.unlink()
                        has_errors = True
                        continue
                    
                    # Show appropriate success message
                    if is_update:
                        msg = f"ℹ️ Updating {uploaded_file.name}"
                        status_message.info(msg)
                    elif is_image:
                        if processed_file.image_metadata:
                            msg_parts = [f"✅ Image processed: {uploaded_file.name}"]
                            
                            # Add vision model results
                            if processed_file.image_metadata.vision_description:
                                msg_parts.append(f"Vision: {processed_file.image_metadata.vision_description[:100]}...")
                            
                            if processed_file.image_metadata.detected_objects:
                                objects = ', '.join(processed_file.image_metadata.detected_objects[:5])
                                if len(processed_file.image_metadata.detected_objects) > 5:
                                    objects += f" and {len(processed_file.image_metadata.detected_objects) - 5} more"
                                msg_parts.append(f"Objects: {objects}")
                            
                            # Add text detection results
                            if processed_file.image_metadata.vision_error and not processed_file.image_metadata.detected_text:
                                msg_parts.append(f"Text detection failed: {processed_file.image_metadata.vision_error}")
                                status_message.warning(' | '.join(msg_parts))
                            elif processed_file.image_metadata.detected_text:
                                text_preview = processed_file.image_metadata.detected_text[:50]
                                if len(processed_file.image_metadata.detected_text) > 50:
                                    text_preview += "..."
                                msg_parts.append(f"Text found: {text_preview}")
                                status_message.success(' | '.join(msg_parts))
                            else:
                                msg_parts.append("No text found")
                                status_message.info(' | '.join(msg_parts))
                        else:
                            msg = f"✅ Image processed: {uploaded_file.name}"
                            status_message.success(msg)
                    else:
                        msg = f"✅ Document processed: {uploaded_file.name}"
                        status_message.success(msg)
                    
                    if close_button.button("✕", key=f"close_{uploaded_file.name}_update_{st.session_state.doc_cache_key}"):
                        status_message.empty()
                        close_button.empty()
                    else:
                        msg = f"✅ {uploaded_file.name} processed successfully"
                        status_message.success(msg)
                        if close_button.button("✕", key=f"close_{uploaded_file.name}_success_{st.session_state.doc_cache_key}"):
                            status_message.empty()
                            close_button.empty()
                    
                    if file_path.exists():
                        file_path.unlink()
                    
                    # Mark file as processed
                    st.session_state[file_key] = True
                        
                except Exception as e:
                    st.session_state.upload_status_messages.append({
                        'type': 'error',
                        'text': f"Error processing {uploaded_file.name}: {str(e)}"
                    })
                    if file_path and file_path.exists():
                        file_path.unlink()
                    has_errors = True
        
        # Display any error messages that occurred during processing
        if has_errors:
            for msg in st.session_state.upload_status_messages:
                if msg['type'] == 'error':
                    st.error(msg['text'])
            # Clear the messages to prevent them from showing again
            st.session_state.upload_status_messages = []
        
        # After all files are processed, increment cache key and rerun
        st.session_state.doc_cache_key += 1
        st.rerun()

    st.write("---")
    
    # Document stats
    st.subheader("📊 Document Stats")
    stats = documents.get('stats', {})
    st.write(f"📚 Documents: {stats.get('total_documents', 0):,}")
    st.write(f"📝 Words: {stats.get('total_words', 0):,}")
    
    st.write("---")
    
    # Document browser
    st.subheader("📑 Documents")
    if not documents.get('documents'):
        st.info("No documents uploaded yet")
    else:
        # Initialize selected docs if not present
        if 'selected_docs' not in st.session_state:
            st.session_state.selected_docs = set()
            
        # Create checkboxes for each document
        for doc in documents.get('documents', {}):
            doc_info = documents['documents'][doc]
            chunks = doc_info.get('total_chunks', 0)
            pages = len(doc_info.get('pages', []))
            words = doc_info.get('word_count', 0)
            
            label = f"{doc} ({chunks} chunks, {pages} pages, {words:,} words)"
            checkbox_key = f"doc_{doc}_{st.session_state.checkbox_key}"
            
            # Handle checkbox state
            if checkbox_key not in st.session_state:
                st.session_state[checkbox_key] = doc in st.session_state.selected_docs
                
            checked = st.checkbox(label, key=checkbox_key)
            
            if checked and doc not in st.session_state.selected_docs:
                st.session_state.selected_docs.add(doc)
            elif not checked and doc in st.session_state.selected_docs:
                st.session_state.selected_docs.discard(doc)
        
        # Show delete button if documents are selected
        if st.session_state.selected_docs:
            st.write("---")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🗑️ Delete Selected", type="primary", key="delete_docs"):
                    for doc in st.session_state.selected_docs:
                        st.session_state.agent.delete_document(doc)
                    st.session_state.selected_docs = set()
                    st.session_state.checkbox_key += 1  # Increment key to force re-render
                    st.success("Selected documents deleted")
                    st.rerun()
            with col2:
                if st.button("Clear", key="clear_docs"):
                    st.session_state.selected_docs = set()
                    st.session_state.checkbox_key += 1  # Increment key to force re-render
                    st.rerun()

# Header
st.title("💬 Chat with your Documents")

st.write("---")

# Add CSS to ensure chat messages use full width
st.markdown("""
<style>
[data-testid="stChatMessageContent"] {
    width: 100% !important;
    max-width: 100% !important;
}

/* Target the message container to ensure full width */
.stChatMessage {
    width: 100% !important;
    max-width: 100% !important;
}

/* Ensure the content inside messages spans full width */
.stMarkdown {
    width: 100% !important;
    max-width: 100% !important;
}

/* Remove any fixed width constraints */
div[data-testid="stChatMessageContent"] > div {
    width: auto !important;
    max-width: none !important;
}
</style>
""", unsafe_allow_html=True)

# Create containers for messages and input
messages_container = st.container()
input_container = st.container()

# Chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display messages
if not st.session_state.messages:
    if not documents.get('documents'):
        st.info("👋 No documents uploaded yet. Upload documents to start asking questions!")
    else:
        st.info("👋 Ready to help! Ask me questions about your uploaded documents.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
prompt = st.chat_input("Ask a question about your documents...")

# Handle input
if prompt:
    if not documents.get('documents'):
        st.warning("Please upload some documents first")
        st.stop()
    
    if st.session_state.selected_model == "-":
        st.error("No models available. Please make sure Ollama is running.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Use selected docs if any are selected, otherwise use all documents
            selected_docs = list(st.session_state.selected_docs) if st.session_state.selected_docs else list(documents.get('documents', {}).keys())
            
            response = st.session_state.agent.chat(
                prompt,
                st.session_state.chat_session,
                model_name=st.session_state.selected_model,
                filter_docs=selected_docs
            )
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
