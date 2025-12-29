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

DOCS_FOLDER = "/mount/src/lasst/documents"
CACHE_FOLDER = os.getenv("CACHE_FOLDER", "./cache")
CHROMA_FOLDER = "./chroma_db"

os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)
os.makedirs(CHROMA_FOLDER, exist_ok=True)

if "collection" not in st.session_state:
    st.session_state.collection = None

if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.active_chat = None

client = chromadb.PersistentClient(path=CHROMA_FOLDER)

collections = client.list_collections()

if collections:
    collection = client.get_collection(
        name=collections[0].name,
        embedding_function=get_embedding_function()
    )
    st.session_state.collection = collection
else:
    with st.spinner("Processing documents..."):
        files = get_files_from_folder()
        if not files:
            st.stop()

        collection = client.create_collection(
            name="biomed_docs",
            embedding_function=get_embedding_function(),
            metadata={"hnsw:space": "cosine"}
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
                    continue
                save_cache(key, info)

            for c in info["chunks"]:
                all_chunks.append(c["content"])
                all_meta[len(all_chunks) - 1] = c["metadata"]
                all_ids.append(f"chunk_{idx}_{len(all_chunks)}")

        for i in range(0, len(all_chunks), 300):
            collection.add(
                documents=all_chunks[i:i+300],
                metadatas=[all_meta[j] for j in range(i, min(i+300, len(all_chunks)))],
                ids=all_ids[i:i+300]
            )

        st.session_state.collection = collection

if not st.session_state.chats:
    cid = f"chat_{uuid.uuid4().hex[:6]}"
    st.session_state.chats[cid] = {
        "title": "New Chat",
        "messages": [],
        "context": []
    }
    st.session_state.active_chat = cid

st.markdown(
    "<h1 style='text-align:center;'>🧬 Biomedical Document Chatbot</h1>",
    unsafe_allow_html=True
)

with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True):
        cid = f"chat_{uuid.uuid4().hex[:6]}"
        st.session_state.chats[cid] = {
            "title": "New Chat",
            "messages": [],
            "context": []
        }
        st.session_state.active_chat = cid
        st.rerun()

    for cid in reversed(list(st.session_state.chats.keys())):
        if st.button(st.session_state.chats[cid]["title"], key=cid, use_container_width=True):
            st.session_state.active_chat = cid
            st.rerun()

chat = st.session_state.chats[st.session_state.active_chat]

for m in chat["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def progressive_query(collection, query):
    collected = []
    seen = set()

    def add(res):
        for d, m in zip(res["documents"][0], res["metadatas"][0]):
            key = (m.get("source"), m.get("page"), d[:60])
            if key not in seen:
                seen.add(key)
                collected.append({"content": d, "metadata": m})

    add(collection.query(query_texts=[query], n_results=12))

    if len(collected) < 6:
        add(collection.query(query_texts=[query], n_results=30))

    q = query.lower()
    if "class a" in q or "katalog a" in q:
        f = [c for c in collected if c["metadata"].get("catalog") == "Katalog A"]
        if f:
            collected = f
    if "class b" in q or "katalog b" in q:
        f = [c for c in collected if c["metadata"].get("catalog") == "Katalog B"]
        if f:
            collected = f

    return collected

def context_is_sufficient(chunks):
    return sum(len(c["content"].split()) for c in chunks) >= 250 or any(
        c["metadata"].get("is_table") == "True" for c in chunks
    )

if query := st.chat_input("Ask anything about the MBE program documents..."):
    chat["messages"].append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            chunks = progressive_query(st.session_state.collection, query)
            if context_is_sufficient(chunks):
                answer = answer_question_with_groq(query, chunks, chat["messages"])
            else:
                answer = "No sufficient information in the available documents."
            st.markdown(answer)

    chat["messages"].append({"role": "assistant", "content": answer})
    chat["context"] = chunks
