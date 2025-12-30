import requests
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import json
import re
import time
import functools
from collections import deque

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    raise ValueError("⚠️ GROQ_API_KEY not set!")

# ========== Rate Limiter ==========
class SimpleRateLimiter:
    """معالج بسيط لـ rate limiting"""
    def __init__(self, max_calls=25, period=60):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque(maxlen=max_calls)
    
    def wait_if_needed(self):
        """انتظر إذا وصلنا للحد الأقصى"""
        now = time.time()
        
        if len(self.calls) >= self.max_calls:
            oldest = self.calls[0]
            if now - oldest < self.period:
                sleep_time = self.period - (now - oldest) + 2
                print(f"⏳ Rate limit: waiting {sleep_time:.0f} seconds...")
                time.sleep(sleep_time)
        
        self.calls.append(now)

limiter = SimpleRateLimiter(max_calls=25, period=60)

# ========== دالة مساعدة: تحديد تعقيد السؤال ==========
def get_dynamic_n_results(query):
    """تحديد عدد chunks المطلوبة حسب تعقيد السؤال"""
    query_lower = query.lower()
    
    # كلمات تدل على أسئلة معقدة
    complex_indicators = [
        'compare', 'all', 'list', 'summarize', 'requirements', 
        'differences', 'between', 'متطلبات', 'جميع', 'قارن',
        'unterschied', 'alle', 'vergleich', 'zusammenfass',
        'if', 'when is', 'deadline', 'calculate'
    ]
    
    # كلمات تدل على أسئلة بسيطة
    simple_indicators = ['what is', 'define', 'who', 'when does', 'where', 'was ist', 'ما هو', 'how many']
    
    if any(ind in query_lower for ind in complex_indicators):
        return 20
    elif any(ind in query_lower for ind in simple_indicators):
        return 8
    
    return 12  # default

# ========== تحسين 1: Query Expansion محسّن ==========
@functools.lru_cache(maxsize=100)
def expand_query(query):
    """توسيع السؤال بذكاء حسب نوعه"""
    
    query_lower = query.lower()
    
    # كشف نوع السؤال وتوليد توسعات مخصصة
    
    # 1. أسئلة عن exam types
    if any(word in query_lower for word in ['exam type', 'prüfungsart', 'examination format', 'exam format']):
        module_name = query.split()[-1] if len(query.split()) > 3 else ""
        return [
            query,
            f"Catalog A {module_name} Prüfungsart",
            f"Catalog B {module_name} examination",
            "SPO Anlage module exam table"
        ]
    
    # 2. أسئلة معقدة عن deadlines مع credits
    if "180 credit" in query_lower or "210 credit" in query_lower:
        if any(word in query_lower for word in ['deadline', 'submit', 'register', 'when']):
            return [
                "4-semester program structure",
                "3-semester program structure", 
                "summer semester start sequence",
                "winter semester start sequence",
                "thesis registration requirements",
                "master thesis duration weeks"
            ]
    
    # 3. أسئلة عن module lists
    if any(word in query_lower for word in ['list all', 'all modules', 'alle module', 'catalog']):
        return [
            query,
            "Wahlpflichtmodule Catalog A complete list",
            "Wahlpflichtmodule Catalog B table",
            "SPO Anlage elective modules"
        ]
    
    # 4. أسئلة مقارنة
    if "compare" in query_lower or "vergleich" in query_lower or "قارن" in query_lower:
        return [
            query,
            "3-semester program structure credits",
            "4-semester program structure credits",
            "differences between programs"
        ]
    
    # 5. توسيع عام بالـ LLM (fallback)
    limiter.wait_if_needed()
    
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
                return [query] + queries[:2]
        except:
            pass
            
    except Exception as e:
        print(f"Query expansion failed: {e}")
    
    return [query]  # fallback

def get_embedding_function():
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
    )

# ========== تحسين 2: Re-ranking محسّن ==========
def rerank_chunks(query, chunks):
    """إعادة ترتيب النتائج مع أولوية للجداول في أسئلة معينة"""
    
    query_lower = query.lower()
    
    # 1. أولوية عالية للجداول في أسئلة القوائم
    if any(word in query_lower for word in ['list', 'all', 'modules', 'catalog', 'elective']):
        table_chunks = [c for c in chunks if c['metadata'].get('is_table') == 'True']
        non_table_chunks = [c for c in chunks if c['metadata'].get('is_table') != 'True']
        
        if table_chunks:
            # الجداول أولاً (معظمها)، ثم النصوص
            return table_chunks[:6] + non_table_chunks[:2]
    
    # 2. أولوية لصفحات SPO في أسئلة الـ regulations
    if any(word in query_lower for word in ['requirement', 'regulation', 'rule', 'voraussetzung', 'regelung']):
        spo_chunks = [c for c in chunks if 'SPO' in c['metadata'].get('source', '')]
        other_chunks = [c for c in chunks if 'SPO' not in c['metadata'].get('source', '')]
        
        if spo_chunks:
            return spo_chunks[:5] + other_chunks[:3]
    
    # 3. Re-ranking بالـ LLM للحالات المعقدة
    if len(chunks) <= 8:
        return chunks  # عدد قليل، لا داعي للـ re-rank
    
    limiter.wait_if_needed()
    
    chunk_texts = [c["content"][:500] for c in chunks[:15]]  # أول 500 حرف من أول 15 chunk
    
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
        
        # استخراج الأرقام
        scores_text = scores_text.replace('[', '').replace(']', '')
        scores = [float(s.strip()) for s in scores_text.split(',') if s.strip()]
        
        if len(scores) == len(chunk_texts):
            # ترتيب تنازلي حسب الدرجة
            scored_chunks = list(zip(chunks[:15], scores))
            sorted_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)
            
            # إرجاع أفضل chunks فوق 4/10
            return [c for c, s in sorted_chunks if s >= 4][:8]
            
    except Exception as e:
        print(f"Reranking failed: {e}")
    
    return chunks[:8]  # fallback

# ========== تحسين 3: Multi-hop Reasoning محسّن ==========
def answer_question_with_groq(query, relevant_chunks, chat_history=None):
    """الإجابة مع دعم قوي للأسئلة المعقدة والـ multi-hop reasoning"""
    
    limiter.wait_if_needed()
    
    # 1. توسيع السؤال (مع cache)
    expanded_queries = expand_query(query)
    
    # 2. إعادة ترتيب النتائج (محسّن)
    reranked_chunks = rerank_chunks(query, relevant_chunks)
    
    # 3. بناء السياق
    context_parts = []
    for i, chunk in enumerate(reranked_chunks, 1):
        source = chunk["metadata"].get("source", "Unknown")
        page = chunk["metadata"].get("page", "N/A")
        is_table = chunk["metadata"].get("is_table", "False")
        content = chunk["content"]
        
        prefix = "📊 TABLE:" if is_table == "True" else "📄 TEXT:"
        context_parts.append(f"[{i}] {prefix} Source: {source} | Page: {page}\n{content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # 4. تلخيص المحادثة
    conversation_summary = ""
    if chat_history and len(chat_history) > 1:
        recent = chat_history[-6:]
        conv_lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            conv_lines.append(f"{role}: {msg['content'][:200]}")
        conversation_summary = "\n".join(conv_lines)
    
    # 5. System prompt محسّن جداً
    system_prompt = """You are a highly accurate assistant for Master Biomedical Engineering (MBE) at Hochschule Anhalt.

CRITICAL MULTI-HOP REASONING RULES:
==================================

1. **Complex Questions Detection:**
   If query contains: "if", "when is deadline", "180/210 credit", "compare", "calculate"
   → This needs MULTI-STEP reasoning

2. **Multi-Step Process (MANDATORY for complex questions):**
   
   Step 1: IDENTIFY what information is needed
   - List ALL pieces of info required (e.g., "program type", "semester sequence", "duration")
   
   Step 2: EXTRACT each piece from sources
   - Search sources for EACH piece separately
   - If a piece is missing, explicitly say which piece is missing
   
   Step 3: COMBINE logically
   - Connect the dots step-by-step
   - Show your reasoning: "Since X is Y, and Z requires Y, then..."
   
   Step 4: CALCULATE/DEDUCE final answer
   - Do any needed calculations
   - State final answer clearly with ALL sources cited

3. **Example of Correct Multi-Hop:**
   
   Query: "If I'm 180-credit bachelor starting summer, when is thesis deadline?"
   
   ✅ CORRECT approach:
   ```
   Step 1: Information needed:
   - 180 credits → which program? (3-sem or 4-sem)
   - Summer start → semester sequence?
   - Thesis is in which semester?
   - Thesis duration?
   
   Step 2: Extract from sources:
   - [Source: SPO §1(2)] 180 credits → 4-semester program
   - [Source: SPO §1(5)] Summer start → sequence: 1-3-2-4
   - [Source: SPO Anlage 1b] Thesis in semester 4
   - [Source: SPO §7(2)] Thesis duration: 20 weeks
   
   Step 3: Combine:
   Starting summer 2025:
   - Semester 1: Summer 2025
   - Semester 3: Winter 2025/26
   - Semester 2: Summer 2026
   - Semester 4: Winter 2026/27 (thesis semester)
   
   Step 4: Calculate deadline:
   Thesis registration: ~January 2027 (end of semester 3)
   Thesis duration: 20 weeks
   Deadline: ~June 2027
   ```
   
   ❌ WRONG approach:
   "No sufficient information" (when all pieces ARE in sources!)

4. **Table Questions:**
   When asked to "list ALL" or "enumerate":
   - Check if sources contain TABLES (marked with 📊)
   - Tables usually have complete lists
   - Extract EVERY item from table
   - Count items and verify completeness
   
   Example: "List all Catalog A modules"
   → Look for 📊 TABLE in sources
   → Extract all 13 modules from table
   → Present as numbered list

5. **Comparison Questions:**
   Structure as side-by-side or table:
   ```
   | Aspect | Option A | Option B |
   |--------|----------|----------|
   | X      | ...      | ...      |
   ```

6. **Strict Sourcing:**
   - Answer ONLY from provided documents
   - If info missing after checking all sources: "No sufficient information in the available documents."
   - NEVER hallucinate or add external knowledge
   - ALWAYS cite: [Source: filename, Page: X]

7. **Language Matching:**
   - Use SAME language as user query
   - English question → English answer
   - German question → German answer
   - Arabic question → Arabic answer

8. **Formatting:**
   - Use numbered lists for steps
   - Use bullet points for multiple items
   - Use tables for comparisons
   - Bold important terms

9. **Follow-ups:**
   Check conversation history first for context"""

    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""CONVERSATION HISTORY:
{conversation_summary if conversation_summary else "No previous conversation"}

EXPANDED SEARCH QUERIES (context):
{', '.join(expanded_queries)}

DOCUMENT SOURCES (📊 = table, 📄 = text):
{context}

CURRENT QUESTION: {query}

INSTRUCTIONS:
- Detect if this is a COMPLEX question (needs multi-hop reasoning)
- If complex: Use step-by-step process (identify → extract → combine → calculate)
- If simple: Answer directly with sources
- For "list ALL": Check for 📊 TABLES and extract completely
- Cite EVERY claim with [Source: file, Page: X]

YOUR ANSWER:"""
            }
        ],
        "temperature": 0.02,  # أقل للدقة الأعلى
        "max_tokens": 2500,  # أكثر للإجابات المعقدة
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=data,
            timeout=90
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        
        return answer
        
    except requests.exceptions.HTTPError as e:
        if "429" in str(e):
            # Rate limit - انتظر وأعد المحاولة
            print("⏳ Rate limit hit, waiting 5 seconds...")
            time.sleep(5)
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json=data,
                    timeout=90
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
            except:
                return "⚠️ Rate limit exceeded. Please wait a moment and try again."
        return f"❌ Error: {str(e)}"
    except Exception as e:
        return f"❌ Error: {str(e)}"
