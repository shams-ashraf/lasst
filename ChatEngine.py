import requests
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import hashlib
import requests
import time
load_dotenv()

def build_answer_cache_key(query: str, context: str) -> str:
    h = hashlib.md5()
    h.update((query + context).encode("utf-8"))
    return h.hexdigest()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    raise ValueError("⚠️ GROQ_API_KEY not set! Please add it to environment variables.")

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
    )

ANSWER_CACHE = {}  # in-memory cache

def answer_question_with_groq(query, relevant_chunks, chat_history=None):
    # ---------- Build context ----------
    context_parts = []
    for chunk in relevant_chunks[:12]:
        source = chunk["metadata"].get("source", "Unknown")
        page = chunk["metadata"].get("page", "N/A")
        content = chunk["content"]
        context_parts.append(
            f"[Source: {source} | Page: {page}]\n{content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # ---------- Cache ----------
    cache_key = build_answer_cache_key(query, context)
    if cache_key in ANSWER_CACHE:
        return ANSWER_CACHE[cache_key]

    # ---------- Conversation summary ----------
    conversation_summary = ""
    if chat_history and len(chat_history) > 1:
        recent = chat_history[-8:]
        conv_lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            conv_lines.append(f"{role}: {msg['content']}")
        conversation_summary = "\n".join(conv_lines)

    # ---------- Payload ----------
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a highly accurate and professional assistant for the Master Biomedical Engineering (MBE) program at Hochschule Anhalt.

CRITICAL RULES:
- Answer EXCLUSIVELY based on the provided document sources or previous conversation history.
- If the question is a follow-up, use the conversation history FIRST.
- If no relevant information exists: Reply exactly "No sufficient information in the available documents."
- Use the SAME language as the user's question.
- Be concise, clear, and professional.
- Always cite sources briefly (document + page).
- NEVER hallucinate or add external knowledge.
"""
            },
            {
                "role": "user",
                "content": f"""
CONVERSATION HISTORY:
{conversation_summary if conversation_summary else "No previous conversation"}

DOCUMENT SOURCES:
{context}

CURRENT QUESTION:
{query}

ANSWER directly and precisely:
"""
            }
        ],
        "temperature": 0.05,
        "max_tokens": 1500,
    }

    # ---------- Retry + Backoff ----------
    for attempt in range(3):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=data,
                timeout=60,
            )

            if response.status_code == 429:
                time.sleep(2 ** attempt)
                continue

            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"].strip()

            ANSWER_CACHE[cache_key] = answer
            return answer

        except Exception:
            if attempt == 2:
                return "⏳ Server is busy. Please try again in a moment."


