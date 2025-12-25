# <DOCUMENT filename="app.py"> (معدل ومحسن ليصل لـ 100% حسب الورقة)
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
from ChatEngine import get_embedding_function, answer_question_with_groq

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

if "collection" not in st.session_state:
    st.session_state.collection = None

if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.active_chat = None

client = chromadb.PersistentClient(path=CHROMA_FOLDER)  # تحسين: استخدم PersistentClient للاستمرارية

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
            st.error("No documents found in the documents folder!")
            st.stop()

        collection = client.create_collection(
            name="biomed_docs",
            embedding_function=get_embedding_function(),
            metadata={"hnsw:space": "cosine"}  # تحسين: cosine similarity لنتائج أفضل
        )

        all_chunks = []
        all_meta = {}
        all_ids = []

        for idx, path in enumerate(files):
            name = os.path.basename(path)
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
                    st.warning(error)
                    continue
                save_cache(key, info)

            for c in info["chunks"]:
                all_chunks.append(c["content"])
                all_meta[len(all_chunks) - 1] = c["metadata"]
                all_ids.append(f"chunk_{idx}_{len(all_chunks)}")

        # إضافة دفعات أصغر لتجنب مشاكل الذاكرة
        batch_size = 300
        for i in range(0, len(all_chunks), batch_size):
            collection.add(
                documents=all_chunks[i:i+batch_size],
                metadatas=[all_meta[j] for j in range(i, min(i+batch_size, len(all_chunks)))],
                ids=all_ids[i:i+batch_size]
            )

        st.session_state.collection = collection
        st.success(f"✅ Processed {len(files)} documents successfully!")

# إنشاء chat جديد إذا لم يكن موجود
if not st.session_state.chats:
    cid = f"chat_{uuid.uuid4().hex[:6]}"
    st.session_state.chats[cid] = {
        "title": "New Chat",
        "messages": [],
        "context": []
    }
    st.session_state.active_chat = cid

# الـ UI الرئيسي
st.markdown("""
<div class="main-card">
    <h1 style='text-align:center;margin:0;'>🧬 Biomedical Document Chatbot</h1>
</div>

<div class="main-card">
    <p style="text-align:center; font-size:1.1rem;">Answers <strong>only</strong> from official documents • Supports English, German & Arabic • Remembers conversation</p>
    <h3 style="color:#00d9ff;">Try these examples:</h3>
    <ul style="font-size:1.05rem;">
        <li>What are the requirements for registering the master's thesis?</li>
        <li>Tell me about the internship requirements</li>
        <li>Summarize the module handbook</li>
        <li>Was sind die Regelungen für die Masterarbeit?</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("# 🧬 BioMed Chat")

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
    for cid in reversed(list(st.session_state.chats.keys())):  # الأحدث فوق
        chat = st.session_state.chats[cid]
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(f"💬 {chat['title'][:35]}...", key=f"open_{cid}", use_container_width=True):
                st.session_state.active_chat = cid
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{cid}"):
                del st.session_state.chats[cid]
                if st.session_state.active_chat == cid:
                    st.session_state.active_chat = next(iter(st.session_state.chats), None)
                st.rerun()

# عرض المحادثة
chat = st.session_state.chats[st.session_state.active_chat]
for m in chat["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
if query := st.chat_input("Ask anything about the MBE program documents..."):
    chat["messages"].append({"role": "user", "content": query})
    if chat["title"] == "New Chat":
        chat["title"] = query[:40] + "..." if len(query) > 40 else query

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents & thinking..."):
            res = st.session_state.collection.query(
                query_texts=[query],
                n_results=12  # زيادة قليلاً لسياق أفضل
            )
            chunks = [{"content": d, "metadata": m} for d, m in zip(res["documents"][0], res["metadatas"][0])]

            answer = answer_question_with_groq(query, chunks, chat["messages"])
            st.markdown(answer)

    chat["messages"].append({"role": "assistant", "content": answer})
    chat["context"] = chunks
