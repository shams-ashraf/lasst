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
    """Build context-aware prompt with chat history"""
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
   
    context_parts = []
    sources_list = []  
    for i, chunk_data in enumerate(relevant_chunks[:10], 1):
        content = chunk_data['content']
        meta = chunk_data['metadata']
        
        source = meta.get('source', 'Unknown')
        page = meta.get('page', 'N/A')
        is_table = meta.get('is_table', 'False')
        table_num = meta.get('table_number', 'N/A')
        
        citation = f"[Source {i}: {source}, Page {page}"
        if is_table == 'True' or is_table == True:
            citation += f", Table {table_num}"
        citation += "]"
        
        context_parts.append(content)
        sources_list.append(citation)
    
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
1. Answer ONLY from provided sources OR previous conversation if it's a follow-up question.
2. For follow-up questions like "summarize", "tell me more", "explain that", or "what about that":
   - Check the conversation history FIRST
   - Summarize or expand on your PREVIOUS answer
3. If user says "summarize that" or "summarize it": Condense your LAST answer (from conversation history)
4. If no relevant info in sources OR history: "No sufficient information in the available documents"
5. Use the SAME language as the question (English/German/Arabic)
6. Be CONCISE - short, direct answers unless asked to elaborate
7. For counting questions: Count precisely and list all items with citations
8. Do NOT explain your thought process.
9. Answer directly and clearly.

Remember: You're helping MBE students understand their program requirements clearly and accurately."""
            },
            {
                "role": "user",
                "content": f"""CONVERSATION HISTORY (use for follow-up questions):
{conversation_summary if conversation_summary else "No previous conversation"}

DOCUMENT SOURCES (use for new factual questions):
{context}

CURRENT QUESTION: {query}

Instructions: 
- If this is a follow-up (summarize/elaborate/that/it), answer from conversation history
- If this is a new question, answer from sources
- Do NOT include your thought process

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
        answer_text = response.json()["choices"][0]["message"]["content"]

        return answer_text
    except Exception as e:
        return f"❌ Error connecting to Groq: {str(e)}"


