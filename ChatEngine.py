import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import requests
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found in environment variables!")

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
    )

def build_conversational_prompt(query, chat_history):
    if not chat_history:
        return query
    recent_history = chat_history[-6:]
    context_lines = []
    for msg in recent_history:
        role = msg['role']
        content = msg['content'][:200]
        if role == 'user':
            context_lines.append(f"Previous Q: {content}")
        else:
            context_lines.append(f"Previous A: {content}")
    history_context = "\n".join(context_lines)
    return f"Conversation context:\n{history_context}\n\nCurrent question: {query}"

def answer_question_with_groq(query, relevant_chunks, chat_history=None):
    if not GROQ_API_KEY:
        return "❌ Please set GROQ_API_KEY in environment variables"
   
    context_parts = [chunk_data['content'] for chunk_data in relevant_chunks[:10]]
    context = "\n\n---\n\n".join(context_parts)
    
    conversation_summary = ""
    if chat_history and len(chat_history) > 0:
        recent = chat_history[-6:]
        conv_lines = []
        for msg in recent:
            role = "User" if msg['role'] == 'user' else "Assistant"
            content_preview = msg['content'][:300]
            conv_lines.append(f"{role}: {content_preview}")
        conversation_summary = "\n".join(conv_lines)
   
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a precise MBE Document Assistant at Hochschule Anhalt specializing in Biomedical Engineering regulations.

CRITICAL RULES:
- Answer based only on provided content or previous conversation
- Ignore source citations in answers
- Use conversation history for follow-ups
- Use same language as question
- Be concise"""
            },
            {
                "role": "user",
                "content": f"""CONVERSATION HISTORY (for follow-ups):
{conversation_summary if conversation_summary else "No previous conversation"}

DOCUMENT CONTENT:
{context}

CURRENT QUESTION: {query}

ANSWER:"""
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
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
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error connecting to Groq: {str(e)}"
