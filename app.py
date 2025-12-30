import streamlit as st
import uuid
import chromadb
import os
from styles import load_custom_css

from DocumentProcessor import (
    get_files_from_folder,
    get_file_hash,
    load_cache,
    extract_pdf_detailed,
    extract_docx_detailed,
    extract_txt_detailed,
    save_cache
)
from ChatEngine import (
    get_embedding_function, 
    answer_question_with_groq,
    get_dynamic_n_results,
    rerank_chunks
)

st.set_page_config(
    page_title="Biomedical Document Chatbot",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_custom_css()

CACHE_FOLDER = os.getenv("CACHE_FOLDER", "./cache")
CHROMA_FOLDER = "./chroma_db"

os.makedirs(CACHE_FOLDER, exist_ok=True)
os.makedirs(CHROMA_FOLDER, exist_ok=True)

# ========== Session State Initialization ==========
if "collection" not in st.session_state:
    st.session_state.collection = None

if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.active_chat = None

if "show_sources" not in st.session_state:
    st.session_state.show_sources = True  # إظهار المصادر افتراضياً

# ========== ChromaDB Setup ==========
client = chromadb.PersistentClient(path=CHROMA_FOLDER)
collections = client.list_collections()

if collections:
    collection = client.get_collection(
        name=collections[0].name,
        embedding_function=get_embedding_function()
    )
    st.session_state.collection = collection
else:
    with st.spinner("📚 Processing documents... This may take a while for the first time."):
        files = get_files_from_folder()
        if not files:
            st.error("❌ No documents found in the documents folder!")
            st.stop()

        collection = client.create_collection(
            name="biomed_docs",
            embedding_function=get_embedding_function(),
            metadata={"hnsw:space": "cosine"}
        )

        all_chunks = []
        all_meta = {}
        all_ids = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, path in enumerate(files):
            name = os.path.basename(path)
            status_text.text(f"Processing: {name}")
            progress_bar.progress((idx + 1) / len(files))
            
            ext = name.split(".")[-1].lower()
            key = f"{get_file_hash(path)}_{ext}"
            cached = load_cache(key)

            if cached:
                info = cached
            else:
                if ext == "pdf":
                    info, error = extract_pdf_detailed(path)
                elif ext in ["doc", "docx"]:
                    info, error = extract_docx_detailed(path)
                elif ext == "txt":
                    info, error = extract_txt_detailed(path)
                else:
                    continue
                if error:
                    st.warning(f"⚠️ {name}: {error}")
                    continue
                save_cache(key, info)

            for c in info["chunks"]:
                all_chunks.append(c["content"])
                all_meta[len(all_chunks) - 1] = c["metadata"]
                all_ids.append(f"chunk_{idx}_{len(all_chunks)}")

        # Batch insertion
        batch_size = 300
        for i in range(0, len(all_chunks), batch_size):
            collection.add(
                documents=all_chunks[i:i+batch_size],
                metadatas=[all_meta[j] for j in range(i, min(i+batch_size, len(all_chunks)))],
                ids=all_ids[i:i+batch_size]
            )

        st.session_state.collection = collection
        progress_bar.progress(1.0)
        status_text.empty()
        st.success(f"✅ Successfully processed {len(files)} documents with {len(all_chunks)} chunks!")

# ========== Initialize Default Chat ==========
if not st.session_state.chats:
    cid = f"chat_{uuid.uuid4().hex[:6]}"
    st.session_state.chats[cid] = {
        "title": "New Chat",
        "messages": [],
        "context": []
    }
    st.session_state.active_chat = cid

# ========== Main UI ==========
st.markdown("""
<div class="main-card">
    <h1 style='text-align:center;margin:0;'>🧬 Biomedical Document Chatbot</h1>
    <p style='text-align:center;color:#888;margin-top:10px;'>Enhanced with Query Expansion & Re-ranking</p>
</div>

<div class="main-card">
    <p style="text-align:center; font-size:1.1rem;">Answers <strong>only</strong> from official documents • Supports English, German & Arabic • Advanced reasoning for complex questions</p>
    <h3 style="color:#00d9ff;">Try these examples:</h3>
    <ul style="font-size:1.05rem;">
        <li>📌 Simple: What are the admission requirements for the MBE program?</li>
        <li>🧩 Complex: If I'm a 180-credit bachelor student starting in summer, when is my thesis deadline?</li>
        <li>📊 Comparison: Compare the 3-semester and 4-semester program structures</li>
        <li>📋 List: List all elective modules in Catalog A with their credits</li>
        <li>🇩🇪 German: Was sind die Regelungen für die Masterarbeit?</li>
        <li>🇸🇦 Arabic: ما هي متطلبات التسجيل في البرنامج؟</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ========== Sidebar ==========
with st.sidebar:
    st.markdown("# 🧬 BioMed Chat")
    
    # Settings
    with st.expander("⚙️ Settings", expanded=False):
        st.session_state.show_sources = st.toggle(
            "Show Sources", 
            value=st.session_state.show_sources,
            help="Display document sources with answers"
        )
        
        # إضافة زر لإعادة بناء قاعدة البيانات
        if st.button("🔄 Rebuild Database", use_container_width=True):
            try:
                client.delete_collection("biomed_docs")
                st.session_state.collection = None
                st.success("✅ Database cleared! Refresh page to rebuild.")
                st.rerun()
            except:
                st.error("❌ Failed to clear database")

    st.divider()

    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        cid = f"chat_{uuid.uuid4().hex[:6]}"
        st.session_state.chats[cid] = {
            "title": "New Chat",
            "messages": [],
            "context": []
        }
        st.session_state.active_chat = cid
        st.rerun()

    st.markdown("### 💬 Your Chats")
    
    # Display chats (most recent first)
    for cid in reversed(list(st.session_state.chats.keys())):
        chat = st.session_state.chats[cid]
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Highlight active chat
            button_type = "primary" if cid == st.session_state.active_chat else "secondary"
            if st.button(
                f"💬 {chat['title'][:35]}..." if len(chat['title']) > 35 else f"💬 {chat['title']}", 
                key=f"open_{cid}", 
                use_container_width=True,
                type=button_type
            ):
                st.session_state.active_chat = cid
                st.rerun()
        
        with col2:
            if st.button("🗑️", key=f"del_{cid}"):
                del st.session_state.chats[cid]
                if st.session_state.active_chat == cid:
                    st.session_state.active_chat = next(iter(st.session_state.chats), None)
                    if st.session_state.active_chat is None:
                        # Create new chat if all deleted
                        new_cid = f"chat_{uuid.uuid4().hex[:6]}"
                        st.session_state.chats[new_cid] = {
                            "title": "New Chat",
                            "messages": [],
                            "context": []
                        }
                        st.session_state.active_chat = new_cid
                st.rerun()

    st.divider()
    st.markdown("### 📊 Statistics")
    chat = st.session_state.chats[st.session_state.active_chat]
    st.metric("Messages", len(chat["messages"]))
    if chat["context"]:
        st.metric("Last Retrieved", f"{len(chat['context'])} chunks")

# ========== Chat Display ==========
chat = st.session_state.chats[st.session_state.active_chat]

for m in chat["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        
        # Display sources if enabled and available
        if m["role"] == "assistant" and st.session_state.show_sources and "sources" in m:
            with st.expander("📚 Sources Used", expanded=False):
                for src in m["sources"]:
                    st.caption(f"📄 **{src['source']}** - Page {src['page']}")

# ========== Chat Input ==========
if query := st.chat_input("Ask anything about the MBE program documents..."):
    chat["messages"].append({"role": "user", "content": query})
    
    # Update chat title
    if chat["title"] == "New Chat":
        chat["title"] = query[:50] + "..." if len(query) > 50 else query

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents & thinking..."):
            # Dynamic retrieval
            n_results = get_dynamic_n_results(query)
            
            status_placeholder = st.empty()
            status_placeholder.caption(f"🔎 Retrieving {n_results} chunks...")
            
            # Initial retrieval
            res = st.session_state.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            chunks = [
                {"content": d, "metadata": m} 
                for d, m in zip(res["documents"][0], res["metadatas"][0])
            ]
            
            status_placeholder.caption(f"⚙️ Re-ranking {len(chunks)} chunks...")
            
            # Re-rank chunks
            try:
                reranked_chunks = rerank_chunks(query, chunks)
            except Exception as e:
                st.warning(f"⚠️ Re-ranking failed: {str(e)}")
                reranked_chunks = chunks[:8]
            
            # Display retrieval stats
            status_placeholder.caption(f"✅ Retrieved {len(chunks)} → Re-ranked to {len(reranked_chunks)} most relevant")
            
            # Generate answer
            answer = answer_question_with_groq(query, reranked_chunks, chat["messages"])
            
            status_placeholder.empty()
            st.markdown(answer)
            
            # Extract sources for display
            sources = []
            seen = set()
            for chunk in reranked_chunks[:5]:  # Top 5 sources
                source_key = f"{chunk['metadata']['source']}_{chunk['metadata']['page']}"
                if source_key not in seen:
                    sources.append({
                        'source': chunk['metadata']['source'],
                        'page': chunk['metadata']['page']
                    })
                    seen.add(source_key)
            
            # Show sources inline if enabled
            if st.session_state.show_sources and sources:
                with st.expander("📚 Sources Used", expanded=False):
                    for src in sources:
                        st.caption(f"📄 **{src['source']}** - Page {src['page']}")

    # Save to chat history
    chat["messages"].append({
        "role": "assistant", 
        "content": answer,
        "sources": sources
    })
    chat["context"] = reranked_chunks

# ========== Footer ==========
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;font-size:0.9rem;'>
    🧬 Biomedical Engineering Chatbot | Powered by ChromaDB + Groq + Llama 3.3<br>
    Enhanced with Query Expansion & Semantic Re-ranking
</div>
""", unsafe_allow_html=True)
