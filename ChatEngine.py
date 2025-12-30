import requests
import time
import hashlib
import os
from chromadb.utils import embedding_functions

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

ANSWER_CACHE = {}  # in-memory cache


def build_answer_cache_key(query: str, context: str) -> str:
    h = hashlib.md5()
    h.update((query + context).encode("utf-8"))
    return h.hexdigest()


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
    )

def answer_question_with_groq(query, relevant_chunks, chat_history=None):
    """
    Guaranteed return: always returns a STRING, never None
    """

    # ---------- Safety checks ----------
    if not GROQ_API_KEY:
        return "❌ GROQ_API_KEY not configured."

    if not relevant_chunks:
        return "No sufficient information in the available documents."

    # ---------- Build context ----------
    context_parts = []
    for chunk in relevant_chunks[:12]:
        try:
            source = chunk.get("metadata", {}).get("source", "Unknown")
            page = chunk.get("metadata", {}).get("page", "N/A")
            content = chunk.get("content", "").strip()

            if content:
                context_parts.append(
                    f"[Source: {source} | Page: {page}]\n{content}"
                )
        except Exception:
            continue

    if not context_parts:
        return "No sufficient information in the available documents."

    context = "\n\n---\n\n".join(context_parts)

    # ---------- Cache ----------
    cache_key = build_answer_cache_key(query, context)
    if cache_key in ANSWER_CACHE:
        return ANSWER_CACHE[cache_key]

    # ---------- Conversation summary ----------
    conversation_summary = ""
    if chat_history and len(chat_history) > 1:
        recent = chat_history[-8:]
        lines = []
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        conversation_summary = "\n".join(lines)

    # ---------- Payload ----------
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a highly accurate and professional assistant for the "
                    "Master Biomedical Engineering (MBE) program at Hochschule Anhalt.\n\n"
                    "CRITICAL RULES:\n"
                    "- Answer EXCLUSIVELY based on the provided document sources or previous conversation history.\n"
                    "- If the question is a follow-up, use the conversation history FIRST.\n"
                    "- If no relevant information exists: Reply exactly "
                    "\"No sufficient information in the available documents.\"\n"
                    "- Use the SAME language as the user's question.\n"
                    "- Be concise, clear, and professional.\n"
                    "- Always cite sources briefly (document + page).\n"
                    "- NEVER hallucinate or add external knowledge."
                )
            },
            {
                "role": "user",
                "content": (
                    f"CONVERSATION HISTORY:\n"
                    f"{conversation_summary if conversation_summary else 'No previous conversation'}\n\n"
                    f"DOCUMENT SOURCES:\n{context}\n\n"
                    f"CURRENT QUESTION:\n{query}\n\n"
                    f"ANSWER directly and precisely:"
                )
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
                json=payload,
                timeout=60,
            )

            if response.status_code == 429:
                time.sleep(2 ** attempt)
                continue

            response.raise_for_status()

            data = response.json()
            answer = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if not answer:
                return "No sufficient information in the available documents."

            ANSWER_CACHE[cache_key] = answer
            return answer

        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)
        except Exception:
            break

    # ---------- Absolute fallback ----------
    return "No sufficient information in the available documents."


