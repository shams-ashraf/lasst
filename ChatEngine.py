import requests
import os
from sentence_transformers import SentenceTransformer
import chromadb
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
                "content": """You are a highly accurate and professional assistant for the Master Biomedical Engineering (MBE) program at Hochschule Anhalt.

CRITICAL RULES:

- Answer EXCLUSIVELY based on the provided document sources or previous conversation history.
- If the question is a follow-up (e.g., "summarize that", "explain more", "what about X"), use the conversation history FIRST.
- If no relevant information exists AFTER reviewing tables, annexes, and headings, reply EXACTLY:
  "No sufficient information in the available documents."
- Use the SAME language as the user's question (English, German, or Arabic).
- Be concise, clear, and professional. Use bullet points or numbering when listing items.
- Always cite sources briefly (e.g., "According to SPO MBE 2024, Anlage 1c...").
- Do NOT infer information that contradicts or goes beyond the document content.
- You MAY infer relationships that are explicitly implied by document structure
  (e.g., table titles, annex headings, section headers that clearly group content).
- NEVER hallucinate, explain your reasoning, or add external knowledge.

SUMMARIZATION RULES:

- For summarization requests of entire documents (e.g., module handbook, SPO):
  - Provide a high-level overview only.
  - Include program duration, total credits, main modules or specializations,
    semester structure, and key regulations.
  - Use bullet points or numbered lists only.
  - Cite multiple pages or annexes where available.

DOMAIN PRIORITIES:

- When asked about Master's thesis registration or regulations, prioritize information from:
  - "94_B14_SPO_MBE_spezifisch_2023-09-27.pdf"
  - "Notes_on_final_theses_at_FB6.pdf"
- For module handbook summaries, list key modules, their credits, and semester distribution if available.
- For counting, listings, or comparisons:
  - Be precise, complete, and avoid assumptions.
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

