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

st.set_page_config(page_title="Biomedical Document Chatbot", page_icon="🧬", layout="wide")

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
cols = client.list_collections()

if cols:
    st.session_state.collection = client.get_collection(
        name=cols[0].name,
        embedding_function=get_embedding_function()
    )
else:
    files = get_files_from_folder()
    col = client.create_collection(
        name="biomed_docs",
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"}
    )
    texts, metas, ids = [], [], []
    for i, p in enumerate(files):
        n = os.path.basename(p)
        ext = n.split(".")[-1].lower()
        k = f"{get_file_hash(p)}_{ext}"
        info = load_cache(k)
        if not info:
            if ext == "pdf":
                info, _ = extract_pdf_detailed(p)
            elif ext in ["doc", "docx"]:
                info, _ = extract_docx_detailed(p)
            else:
                info, _ = extract_txt_detailed(p)
            save_cache(k, info)
        for c in info["chunks"]:
            texts.append(c["content"])
            metas.append(c["metadata"])
            ids.append(f"{i}_{len(ids)}")
    for i in range(0, len(texts), 300):
        col.add(
            documents=texts[i:i+300],
            metadatas=metas[i:i+300],
            ids=ids[i:i+300]
        )
    st.session_state.collection = col

if not st.session_state.chats:
    cid = f"chat_{uuid.uuid4().hex[:6]}"
    st.session_state.chats[cid] = {"title": "New Chat", "messages": []}
    st.session_state.active_chat = cid

st.markdown("<h1 style='text-align:center;'>🧬 Biomedical Document Chatbot</h1>", unsafe_allow_html=True)

chat = st.session_state.chats[st.session_state.active_chat]

for m in chat["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def retrieve(collection, query):
    res = collection.query(query_texts=[query], n_results=25)
    return [{"content": d, "metadata": m} for d, m in zip(res["documents"][0], res["metadatas"][0])]

def sufficient(chunks):
    return sum(len(c["content"].split()) for c in chunks) >= 300 or any(
        c["metadata"].get("is_table") == "True" for c in chunks
    )

if q := st.chat_input("Ask anything about the MBE program documents..."):
    chat["messages"].append({"role": "user", "content": q})
    with st.chat_message("assistant"):
        chunks = retrieve(st.session_state.collection, q)
        if sufficient(chunks):
            a = answer_question_with_groq(q, chunks, chat["messages"])
        else:
            a = "No sufficient information in the available documents."
        st.markdown(a)
    chat["messages"].append({"role": "assistant", "content": a})
