import os
import requests
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
    )

def answer_question_with_groq(query, chunks, history=None):
    ctx = "\n\n---\n\n".join(
        f"[Source: {c['metadata']['source']} | Page: {c['metadata']['page']}]\n{c['content']}"
        for c in chunks[:12]
    )
    hist = ""
    if history and len(history) > 1:
        hist = "\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in history[-8:]
        )
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are an academic-grade assistant for official university regulations and module documents.

You must answer ONLY using the provided document excerpts and conversation history.

Interpret the user's intent carefully, even if wording is incomplete, informal, or imprecise, but only when the document structure clearly supports such interpretation (e.g., annex titles, table headings, catalogs).

Before concluding that information is missing, exhaustively consider all provided excerpts, including tables, annexes, and section groupings.

You may infer relationships that are explicitly implied by document organization, but you must not invent facts, names, rules, credits, or classifications.

If and only if the documents do not support an answer after this analysis, reply exactly:
No sufficient information in the available documents.

Always answer in the same language as the user.
Use an academic, factual tone.
Reference document sources when possible."""
            },
            {
                "role": "user",
                "content": f"CONVERSATION HISTORY:\n{hist if hist else 'None'}\n\nDOCUMENT SOURCES:\n{ctx}\n\nQUESTION:\n{query}\n\nANSWER:"
            }
        ],
        "temperature": 0.05,
        "max_tokens": 1500
    }
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json=data,
        timeout=60
    )
    return r.json()["choices"][0]["message"]["content"].strip()
