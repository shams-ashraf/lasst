"""
UI Manager - Streamlit Interface Only
"""
import streamlit as st
import os
import shutil

class UIManager:
    def __init__(self, doc_processor, chat_engine):
        self.doc_processor = doc_processor
        self.chat_engine = chat_engine
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'processed' not in st.session_state:
            st.session_state.processed = False
            st.session_state.files_data = {}
            st.session_state.messages = []
            st.session_state.current_context = []
    
    def render_header(self):
        """Render main header"""
        st.markdown("""
        <div class="main-card">
            <h1 style='text-align: center; margin: 0;'>🎓 MBE Document Assistant</h1>
            <p style='text-align: center; margin-top: 10px;'>RAG Chatbot for Biomedical Engineering at Hochschule Anhalt</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with document management"""
        with st.sidebar:
            st.markdown("### 📚 Document Management")
            
            # Show available files
            available_files = self.doc_processor.get_available_files()
            
            if not available_files:
                st.warning("⚠️ No documents found")
                st.info(f"📁 Add files to: {self.doc_processor.docs_folder}")
            else:
                st.success(f"✅ {len(available_files)} document(s)")
                with st.expander("📂 Files", expanded=False):
                    for file in available_files:
                        st.write(f"• {os.path.basename(file)}")
            
            st.markdown("---")
            
            # Process button
            if available_files and st.button("🚀 Process Documents", 
                                            type="primary", 
                                            use_container_width=True):
                self._process_documents(available_files)
            
            # Show statistics if processed
            if st.session_state.processed:
                self._render_statistics()
                self._render_action_buttons()
    
    def _process_documents(self, available_files):
        """Process all documents"""
        with st.spinner("Processing documents..."):
            files_data = {}
            all_chunks = []
            all_metadata = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, filepath in enumerate(available_files):
                filename = os.path.basename(filepath)
                status_text.text(f"Processing: {filename}...")
                
                # Process file
                file_info, error = self.doc_processor.process_file(filepath)
                
                if error:
                    st.error(f"❌ {filename}: {error}")
                    continue
                
                files_data[filename] = file_info
                
                # Collect chunks and metadata
                for chunk_obj in file_info['chunks']:
                    if isinstance(chunk_obj, dict):
                        all_chunks.append(chunk_obj['content'])
                        all_metadata.append(chunk_obj['metadata'])
                    else:
                        # Fallback for old format
                        all_chunks.append(chunk_obj)
                        all_metadata.append({
                            "source": filename,
                            "page": "N/A",
                            "is_table": "False",
                            "table_number": "N/A"
                        })
                
                progress_bar.progress((idx + 1) / len(available_files))
            
            status_text.text("Building search index...")
            
            # Initialize chat engine with all chunks
            if all_chunks:
                success = self.chat_engine.initialize_collection(
                    all_chunks, all_metadata
                )
                
                if success:
                    st.session_state.files_data = files_data
                    st.session_state.processed = True
                    st.session_state.messages = []
                    
                    status_text.empty()
                    st.success("✅ Processing completed!")
                    st.balloons()
                else:
                    st.error("❌ Failed to initialize search index")
    
    def _render_statistics(self):
        """Render processing statistics"""
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        
        total_chunks = sum(
            len(info['chunks']) 
            for info in st.session_state.files_data.values()
        )
        total_tables = sum(
            info['total_tables'] 
            for info in st.session_state.files_data.values()
        )
        
        st.metric("Files", len(st.session_state.files_data))
        st.metric("Chunks", total_chunks)
        st.metric("Tables", total_tables)
    
    def _render_action_buttons(self):
        """Render action buttons"""
        st.markdown("---")
        
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = []
            st.session_state.current_context = []
            st.rerun()
        
        if st.button("🗑️ Clear Cache & Reprocess"):
            cache_folder = self.doc_processor.cache_folder
            if os.path.exists(cache_folder):
                shutil.rmtree(cache_folder)
                os.makedirs(cache_folder)
            
            st.session_state.processed = False
            st.session_state.files_data = {}
            st.session_state.messages = []
            st.success("✅ Cache cleared! Click 'Process Documents'.")
            st.rerun()
    
    def render_chat_interface(self):
        """Render main chat interface"""
        if not st.session_state.processed:
            self._render_welcome_screen()
            return
        
        st.markdown("### 💬 Chat with Documents")
        
        # Display chat history
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.markdown(
                    f'<div class="chat-message user-message">'
                    f'👤 <b>You:</b> {content}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message assistant-message">'
                    f'🤖 <b>Assistant:</b><br>{content}</div>',
                    unsafe_allow_html=True
                )
        
        # Chat input
        query = st.chat_input("Ask anything about your documents...")
        
        if query:
            self._handle_user_query(query)
        
        # Show sources
        if st.session_state.current_context:
            self._render_sources()
    
    def _handle_user_query(self, query: str):
        """Handle user query"""
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": query
        })
        
        with st.spinner("Thinking..."):
            # Get answer from chat engine
            answer, relevant_chunks = self.chat_engine.answer_question(
                query, 
                st.session_state.messages
            )
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
            
            # Store context for sources display
            st.session_state.current_context = relevant_chunks
        
        st.rerun()
    
    def _render_sources(self):
        """Render sources in expander"""
        with st.expander("📄 View Sources", expanded=False):
            for idx, chunk_data in enumerate(
                st.session_state.current_context[:5], 1
            ):
                meta = chunk_data['metadata']
                
                source = meta.get('source', 'Unknown')
                page = meta.get('page', 'N/A')
                is_table = meta.get('is_table', 'False')
                table_num = meta.get('table_number', 'N/A')
                
                citation_info = f"📄 **Source {idx}**: {source} | Page {page}"
                if is_table in ['True', True]:
                    citation_info += f" | Table {table_num}"
                
                st.markdown(citation_info)
                st.markdown(
                    f'<div class="chunk-display">'
                    f'{chunk_data["content"][:500]}...</div>',
                    unsafe_allow_html=True
                )
                st.markdown("---")
    
    def _render_welcome_screen(self):
        """Render welcome screen when not processed"""
        st.markdown("""
        ### 🎯 Features:
        - **Precise Citations**: Every answer includes file + page + table references
        - **Conversational**: Ask follow-up questions naturally
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
    
    def render(self):
        """Main render method"""
        self.render_header()
        self.render_sidebar()
        self.render_chat_interface()
