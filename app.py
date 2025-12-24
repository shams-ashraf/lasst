import streamlit as st
import uuid
import chromadb
import os
from styles import load_custom_css
from DocumentProcessor import get_files_from_folder,get_file_hash,load_cache,extract_txt_detailed,extract_pdf_detailed,extract_docx_detailed,save_cache
from ChatEngine import get_embedding_function,answer_question_with_groq

load_custom_css()

if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.files_data = {}
    st.session_state.collection = None
    st.session_state.messages = []  # Chat history
    st.session_state.current_context = []  # للـ conversational flow

# Configuration

PDF_PASSWORD = "mbe2025"
DOCS_FOLDER = "/mount/src/test/documents"
CACHE_FOLDER = os.getenv("CACHE_FOLDER", "./cache")

os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

# Main UI
st.markdown("""
<div class="main-card">
    <h1 style='text-align: center; margin: 0;'>🧬 Biomedical Document Chatbot</h1>
    <p style='text-align: center; margin-top: 10px;'>RAG Chatbot for Biomedical Engineering at Hochschule Anhalt</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for document management
with st.sidebar:
    st.markdown("#🧬 BioMed Doc Chat")
    
    available_files = get_files_from_folder()
    if not available_files:
        st.warning(f"⚠️ No documents found")
        st.info(f"📁 Add files to: {DOCS_FOLDER}")
    else:
        st.success(f"✅ {len(available_files)} document(s)")
        with st.expander("📂 Files", expanded=False):
            for file in available_files:
                st.write(f"• {os.path.basename(file)}")
    
    st.markdown("---")
    
    if available_files and st.button("🚀 Process Documents", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            files_data = {}
            all_chunks = []
            all_metadata = []
           
            client = chromadb.Client()
            collection_name = f"docs_{uuid.uuid4().hex[:8]}"
            collection = client.create_collection(
                name=collection_name,
                embedding_function=get_embedding_function()
            )
           
            progress_bar = st.progress(0)
            status_text = st.empty()
           
            for idx, filepath in enumerate(available_files):
                filename = os.path.basename(filepath)
                file_ext = filename.split('.')[-1].lower()
               
                status_text.text(f"Processing: {filename}...")
               
                file_hash = get_file_hash(filepath)
                cache_key = f"{file_hash}_{file_ext}"
                cached_data = load_cache(cache_key)
               
                if cached_data:
                    st.info(f"📦 Cached: {filename}")
                    file_info = cached_data
                    error = None
                else:
                    if file_ext == 'pdf':
                        file_info, error = extract_pdf_detailed(filepath)
                    elif file_ext in ['docx', 'doc']:
                        file_info, error = extract_docx_detailed(filepath)
                    elif file_ext == 'txt':
                        file_info, error = extract_txt_detailed(filepath)
                    else:
                        error = "Unsupported file type"
                        file_info = None
                   
                    if error:
                        st.error(f"❌ {filename}: {error}")
                        continue
                   
                    save_cache(cache_key, file_info)
                    st.success(f"💾 Cached: {filename}")
               
                files_data[filename] = file_info
               
                # Add chunks with full metadata
                for chunk_obj in file_info['chunks']:
                    # Handle both dict and string formats
                    if isinstance(chunk_obj, dict):
                        all_chunks.append(chunk_obj['content'])
                        all_metadata.append(chunk_obj['metadata'])
                    else:
                        # Fallback for old cached data
                        all_chunks.append(chunk_obj)
                        all_metadata.append({
                            "source": filename, 
                            "page": "N/A",
                            "is_table": "False",
                            "table_number": "N/A"
                        })
               
                progress_bar.progress((idx + 1) / len(available_files))
           
            status_text.text("Building search index...")
           
            if all_chunks:
                batch_size = 500
                for i in range(0, len(all_chunks), batch_size):
                    batch = all_chunks[i:i+batch_size]
                    metadata_batch = all_metadata[i:i+batch_size]
                    collection.add(
                        documents=batch,
                        ids=[f"chunk_{i+j}" for j in range(len(batch))],
                        metadatas=metadata_batch
                    )
           
            st.session_state.files_data = files_data
            st.session_state.collection = collection
            st.session_state.processed = True
            st.session_state.messages = []  # Reset chat
           
            status_text.empty()
            st.success("✅ Processing completed!")
            st.balloons()
    
    if st.session_state.processed:
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        total_chunks = sum(len(info['chunks']) for info in st.session_state.files_data.values())
        total_tables = sum(info['total_tables'] for info in st.session_state.files_data.values())
        
        st.metric("Files", len(st.session_state.files_data))
        st.metric("Chunks", total_chunks)
        st.metric("Tables", total_tables)
        
        st.markdown("---")
        
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = []
            st.session_state.current_context = []
            st.rerun()
        
        if st.button("🗑️ Clear Cache & Reprocess"):
            import shutil
            if os.path.exists(CACHE_FOLDER):
                shutil.rmtree(CACHE_FOLDER)
                os.makedirs(CACHE_FOLDER)
            st.session_state.processed = False
            st.session_state.files_data = {}
            st.session_state.collection = None
            st.session_state.messages = []
            st.success("✅ Cache cleared! Click 'Process Documents' to reprocess.")
            st.rerun()

# Main chat interface
if st.session_state.processed:
    st.markdown("### 💬 Chat with Documents")
    
    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f'<div class="chat-message user-message">👤 <b>You:</b> {content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message">🤖 <b>Assistant:</b><br>{content}</div>', unsafe_allow_html=True)
    
    # Chat input
    query = st.chat_input("Ask anything about your documents...")
    
    if query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.spinner("Thinking..."):
            # Search with metadata
            results = st.session_state.collection.query(
                query_texts=[query],
                n_results=10
            )
            
            # Build chunk objects with metadata
            relevant_chunks = []
            for content, metadata in zip(results["documents"][0], results["metadatas"][0]):
                relevant_chunks.append({
                    'content': content,
                    'metadata': metadata
                })
            
            # Generate answer with chat history
            answer = answer_question_with_groq(query, relevant_chunks, st.session_state.messages)
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Store context for follow-ups
            st.session_state.current_context = relevant_chunks
        
        st.rerun()
    
    # Show sources in expander
    if st.session_state.current_context:
        with st.expander("📄 View Sources", expanded=False):
            for idx, chunk_data in enumerate(st.session_state.current_context[:5], 1):
                meta = chunk_data['metadata']
                
                # Handle string metadata
                source = meta.get('source', 'Unknown')
                page = meta.get('page', 'N/A')
                is_table = meta.get('is_table', 'False')
                table_num = meta.get('table_number', 'N/A')
                
                citation_info = f"📄 **Source {idx}**: {source} | Page {page}"
                if is_table == 'True' or is_table == True:
                    citation_info += f" | Table {table_num}"
                
                st.markdown(citation_info)
                st.markdown(f'<div class="chunk-display">{chunk_data["content"][:500]}...</div>', unsafe_allow_html=True)
                st.markdown("---")

else:
    st.info("👈 Click 'Process Documents' in the sidebar to start!")
    
    st.markdown("""
    ### 🎯 Features:
    - **Precise Citations**: Every answer includes file + page + table references
    - **Conversational**: Ask follow-up questions naturally ("summarize that", "tell me more")
    - **Smart Context**: Understands when you refer to previous answers
    - **Multi-language**: Supports English, German, and Arabic
    - **Fast**: Cached processing for instant responses
    - **MBE-Specific**: Optimized for biomedical engineering regulations
    
    ### 📋 Supported Documents:
    - 📄 Study & Examination Regulations (SPO)
    - 📚 Module Handbook
    - 📝 Guide for Writing Scientific Papers
    - 📃 Notes on Bachelor/Master Theses
    - ✍️ Scientific Writing Guidelines
    
    ### 💡 Example Questions:
    - "How many modules in semester 1?"
    - "What are the thesis requirements?"
    - "Tell me about the internship" → then "summarize that"
    - "Compare exam types in SPO"
    """)
