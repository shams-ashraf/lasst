import requests
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import json

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    raise ValueError("⚠️ GROQ_API_KEY not set!")

def get_embedding_function():
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
    )

# ========== تحسين 1: Query Expansion ==========
def expand_query(query):
    """توسيع السؤال لاستخراج معلومات أفضل"""
    
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a query expansion expert. Generate 2-3 related search queries 
that would help find relevant information. Return ONLY a JSON array of strings.

Example:
User: "When do I submit my thesis if I start in summer?"
Output: ["thesis submission deadline", "master thesis duration weeks", "summer semester start dates"]"""
            },
            {
                "role": "user",
                "content": f"Original query: {query}\nGenerate search variations:"
            }
        ],
        "temperature": 0.3,
        "max_tokens": 150,
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=data,
            timeout=30
        )
        response.raise_for_status()
        expanded = response.json()["choices"][0]["message"]["content"].strip()
        
        # محاولة استخراج JSON
        try:
            queries = json.loads(expanded)
            if isinstance(queries, list):
                return [query] + queries[:2]  # السؤال الأصلي + 2 توسعات
        except:
            pass
            
    except Exception as e:
        print(f"Query expansion failed: {e}")
    
    return [query]  # fallback

# ========== تحسين 2: Re-ranking ==========
def rerank_chunks(query, chunks):
    """إعادة ترتيب النتائج حسب الأهمية الحقيقية"""
    
    chunk_texts = [c["content"][:500] for c in chunks]  # أول 500 حرف
    
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a relevance scorer. For each chunk, rate its relevance to the query from 0-10.
Return ONLY a JSON array of numbers (one per chunk).

Example:
Query: "thesis submission requirements"
Chunks: ["About module structure...", "Thesis must be submitted...", "Course schedule..."]
Output: [2, 9, 1]"""
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nChunks:\n" + "\n---\n".join([f"[{i}] {t}" for i, t in enumerate(chunk_texts)])
            }
        ],
        "temperature": 0.1,
        "max_tokens": 100,
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=data,
            timeout=30
        )
        response.raise_for_status()
        scores_text = response.json()["choices"][0]["message"]["content"].strip()
        
        scores = json.loads(scores_text)
        if isinstance(scores, list) and len(scores) == len(chunks):
            # ترتيب تنازلي حسب الدرجة
            ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
            return [c for c, s in ranked if s >= 4][:8]  # أفضل 8 chunks فوق 4/10
            
    except Exception as e:
        print(f"Reranking failed: {e}")
    
    return chunks[:8]  # fallback

# ========== تحسين 3: Multi-hop Reasoning ==========
def answer_question_with_groq(query, relevant_chunks, chat_history=None):
    """الإجابة مع دعم الأسئلة المعقدة"""
    
    # 1. توسيع السؤال
    expanded_queries = expand_query(query)
    
    # 2. إعادة ترتيب النتائج
    reranked_chunks = rerank_chunks(query, relevant_chunks)
    
    # 3. بناء السياق
    context_parts = []
    for i, chunk in enumerate(reranked_chunks, 1):
        source = chunk["metadata"].get("source", "Unknown")
        page = chunk["metadata"].get("page", "N/A")
        content = chunk["content"]
        context_parts.append(f"[{i}] Source: {source} | Page: {page}\n{content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # 4. تلخيص المحادثة
    conversation_summary = ""
    if chat_history and len(chat_history) > 1:
        recent = chat_history[-6:]
        conv_lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            conv_lines.append(f"{role}: {msg['content'][:200]}")  # اختصار للسرعة
        conversation_summary = "\n".join(conv_lines)
    
    # 5. System prompt محسّن
    system_prompt = """You are a highly accurate assistant for Master Biomedical Engineering (MBE) at Hochschule Anhalt.

CRITICAL RULES:
1. **Multi-step reasoning:** If the question requires connecting multiple facts, break it down:
   - Step 1: Identify what information is needed
   - Step 2: Extract each piece from sources
   - Step 3: Combine logically
   - Step 4: Provide final answer

2. **Complex questions:** For questions like "If X, then when Y?":
   - Extract condition X from sources
   - Find rule/timeline for Y
   - Calculate/deduce the answer
   - Cite ALL relevant sources

3. **Comparison questions:** For "compare A vs B":
   - List criteria for A with sources
   - List criteria for B with sources
   - Present side-by-side or bullet points

4. **Strict sourcing:**
   - Answer ONLY from provided documents
   - If insufficient info: "No sufficient information in the available documents."
   - NEVER hallucinate or add external knowledge

5. **Language matching:** Use the SAME language as user (English/German/Arabic)

6. **Formatting:**
   - Use bullet points for lists
   - Use step-by-step for complex answers
   - Always cite sources (e.g., "[Source: SPO_MBE, Page 3]")

7. **Follow-ups:** Check conversation history first for context"""

    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""CONVERSATION HISTORY:
{conversation_summary if conversation_summary else "No previous conversation"}

EXPANDED SEARCH QUERIES (for context):
{', '.join(expanded_queries)}

DOCUMENT SOURCES (ranked by relevance):
{context}

CURRENT QUESTION: {query}

INSTRUCTIONS:
- If this is a complex question, use step-by-step reasoning
- Cite specific sources for each claim
- Be precise and complete
- If information is missing, state clearly

YOUR ANSWER:"""
            }
        ],
        "temperature": 0.05,
        "max_tokens": 2000,  # زيادة للإجابات المعقدة
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=data,
            timeout=90  # زيادة timeout
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        
        # إضافة معلومات debug (اختياري)
        debug_info = f"\n\n---\n*Searched: {len(expanded_queries)} queries | Retrieved: {len(reranked_chunks)} relevant chunks*"
        
        return answer  # أو answer + debug_info إذا أردت
        
    except Exception as e:
        return f"❌ Error: {str(e)}"
