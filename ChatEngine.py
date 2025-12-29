import requests
import os
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    raise ValueError("⚠️ GROQ_API_KEY not set! Please add it to environment variables.")

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
    )

def answer_question_with_groq(query, relevant_chunks, chat_history=None):
    context_parts = []
    for i, chunk in enumerate(relevant_chunks[:12], 1):
        source = chunk["metadata"].get("source", "Unknown")
        page = chunk["metadata"].get("page", "N/A")
        content = chunk["content"]
        context_parts.append(f"[Source: {source} | Page: {page}]\n{content}")

    context = "\n\n---\n\n".join(context_parts)

    conversation_summary = ""
    if chat_history and len(chat_history) > 1:
        recent = chat_history[-8:]
        conv_lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            conv_lines.append(f"{role}: {msg['content']}")
        conversation_summary = "\n".join(conv_lines)

    data = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                # 🔥 التعديل الوحيد هنا
                "content": """You are an academic assistant for the Master Biomedical Engineering (MBE) program at Hochschule Anhalt.

CORE PRINCIPLES (MANDATORY):

1. Interpret the user’s question semantically, not literally.
   - Identify the intended academic concept even if the wording is incomplete, informal, or imprecise.
   - Resolve equivalent academic terms implicitly (e.g. classifications, catalogs, annexes).

2. Always search across:
   - Main document text
   - Tables
   - Annexes (Anlagen)
   - Study and examination plans
   - Module catalogs

3. Prioritize sources in this strict order:
   - Official study and examination regulations (SPO)
   - Annexes and module catalogs
   - Study plans and structured tables
   - Module handbook descriptions

4. Use conversation history ONLY if the question is a follow-up.

5. Answer strictly and exclusively based on the provided document sources.
   - Do NOT infer, guess, generalize, or use external knowledge.

6. Fallback rule:
   - Reply exactly:
     "No sufficient information in the available documents."
   - ONLY after confirming that all relevant sections, tables, and annexes were checked.

7. Academic rigor:
   - Be concise and structured.
   - Use bullet points or numbered lists.
   - Always cite the document name and page number.

8. Language:
   - Answer in the SAME language as the user.

9. Never explain your reasoning.
"""
            },
            {
                "role": "user",
                "content": f"""CONVERSATION HISTORY (for follow-ups only):
{conversation_summary if conversation_summary else "No previous conversation"}

DOCUMENT SOURCES:
{context}

CURRENT QUESTION: {query}

ANSWER directly and precisely:"""
            }
        ],
        "temperature": 0.05,
        "max_tokens": 1500,
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"
