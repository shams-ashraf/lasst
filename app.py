import streamlit as st
import uuid
import chromadb
import os
from styles import load_custom_css
from DocumentProcessor import (
    get_files_from_folder,
    get_file_hash,
    load_cache,
    extract_txt_detailed,
    extract_pdf_detailed,
    extract_docx_detailed,
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

if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.collection = None

if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.active_chat = None

PDF_PASSWORD = "mbe2025"
DOCS_FOLDER = "/mount/src/test/documents"
CACHE_FOLDER = os.getenv("CACHE_FOLDER", "./cache")

os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

if not st.session_state.processed:
    with st.spinner("📚 Processing documents..."):
        files = get_files_from_folder()
        if not files:
            st.error("No documents found")
            st.stop()

        client = chromadb.Client()
        collection = client.create_collection(
            name=f"docs_{uuid.uuid4().hex[:8]}",
            embedding_function=get_embedding_function()
        )

        all_chunks = []
        all_meta = {}

        for path in files:
            name = os.path.basename(path)
            ext = name.split(".")[-1].lower()
            key = f"{get_file_hash(path)}_{ext}"
            cached = load_cache(key)

            if cached:
                info = cached
            else:
                if ext == "pdf":
                    info, _ = extract_pdf_detailed(path)
                elif ext in ["doc", "docx"]:
                    info, _ = extract_docx_detailed(path)
                elif ext == "txt":
                    info, _ = extract_txt_detailed(path)
                else:
                    continue
                save_cache(key, info)

            for c in info["chunks"]:
                if isinstance(c, dict):
                    all_chunks.append(c["content"])
                    all_meta[len(all_chunks) - 1] = c["metadata"]
                else:
                    all_chunks.append(c)
                    all_meta[len(all_chunks) - 1] = {
                        "source": name,
                        "page": "N/A",
                        "is_table": "False",
                        "table_number": "N/A"
                    }

        for i in range(0, len(all_chunks), 500):
            collection.add(
                documents=all_chunks[i:i+500],
                metadatas=[all_meta[j] for j in range(i, min(i+500, len(all_chunks)))],
                ids=[f"chunk_{j}" for j in range(i, min(i+500, len(all_chunks)))]
            )

        st.session_state.collection = collection
        st.session_state.processed = True

        if not st.session_state.chats:
            cid = f"chat_{uuid.uuid4().hex[:6]}"
            st.session_state.chats[cid] = {
                "title": "New Chat",
                "messages": [],
                "context": []
            }
            st.session_state.active_chat = cid

    st.rerun()

st.markdown("""
<div class="main-card">
    <h1 style='text-align: center; margin: 0;'>🧬 Biomedical Document Chatbot</h1>
    <p style='text-align: center; margin-top: 10px;'>RAG Chatbot for Biomedical Engineering at Hochschule Anhalt</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# 🧬 BioMed Doc Chat")

    files = get_files_from_folder()
    if files:
        st.success(f"✅ {len(files)} document(s)")
        with st.expander("📂 Files"):
            for f in files:
                st.write(f"• {os.path.basename(f)}")

    st.markdown("---")

    st.markdown("### ℹ️ About")
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

    st.markdown("---")
    st.markdown("### 💬 Chats")

    if st.button("🆕 New Chat", use_container_width=True):
        cid = f"chat_{uuid.uuid4().hex[:6]}"
        st.session_state.chats[cid] = {
            "title": "New Chat",
            "messages": [],
            "context": []
        }
        st.session_state.active_chat = cid
        st.rerun()

    for cid in list(st.session_state.chats.keys()):
        col1, col2 = st.columns([5, 1])
        title = st.session_state.chats[cid]["title"]
        with col1:
            if st.button(f"💬 {title}", key=f"open_{cid}", use_container_width=True):
                st.session_state.active_chat = cid
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{cid}", use_container_width=True):
                del st.session_state.chats[cid]
                if st.session_state.active_chat == cid:
                    st.session_state.active_chat = None
                st.rerun()

if st.session_state.active_chat is None:
    st.info("👈 Start a new chat")
    st.stop()

chat = st.session_state.chats[st.session_state.active_chat]
messages = chat["messages"]

st.markdown("### 💬 Chat")

for m in messages:
    if m["role"] == "user":
        st.markdown(f"<div class='chat-message user-message'>👤 {m['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-message assistant-message'>🤖<br>{m['content']}</div>", unsafe_allow_html=True)

query = st.chat_input("Ask anything about your documents...")

if query:
    messages.append({"role": "user", "content": query})
    if chat["title"] == "New Chat":
        chat["title"] = query[:40]

    with st.spinner("Thinking..."):
        res = st.session_state.collection.query(
            query_texts=[query],
            n_results=10
        )

        chunks = []
        for d, m in zip(res["documents"][0], res["metadatas"][0]):
            chunks.append({"content": d, "metadata": m})

        answer = answer_question_with_groq(query, chunks, messages)
        messages.append({"role": "assistant", "content": answer})
        chat["context"] = chunks

    st.rerun()

if chat["context"]:
    with st.expander("📄 Sources"):
        for i, c in enumerate(chat["context"][:5], 1):
            meta = c["metadata"]
            src = meta.get("source", "Unknown")
            page = meta.get("page", "N/A")
            table = meta.get("table_number", "N/A")
            is_table = meta.get("is_table", "False")
            line = f"📄 Source {i}: {src} | Page {page}"
            if is_table in [True, "True"]:
                line += f" | Table {table}"
            st.markdown(line)
            st.markdown(f"<div class='chunk-display'>{c['content'][:500]}...</div>", unsafe_allow_html=True)
