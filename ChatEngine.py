import requests
import os
import time
import hashlib
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    raise ValueError("⚠️ GROQ_API_KEY not set! Please add it to environment variables.")

GROQ_RATE_LIMIT_UNTIL = 0


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
    )


class ImprovedChatEngine:
    def __init__(self):
        self.query_cache = {}
        self.max_cache_size = 100

    def calculate_tokens(self, text):
        return int(len(text.split()) * 1.3)

    def fit_context_to_window(self, chunks, max_tokens=6000):
        context_parts = []
        used_chunks = []
        total_tokens = 0

        for chunk in chunks:
            source = chunk["metadata"].get("source", "Unknown")
            page = chunk["metadata"].get("page", "N/A")
            content = chunk["content"]

            chunk_text = f"[Source: {source} | Page: {page}]\n{content}"
            chunk_tokens = self.calculate_tokens(chunk_text)

            if total_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(chunk_text)
            used_chunks.append({
                "source": source,
                "page": page,
                "content": content
            })
            total_tokens += chunk_tokens

        return "\n\n---\n\n".join(context_parts), used_chunks

    def answer_question_with_groq(self, query, relevant_chunks, chat_history=None):
        global GROQ_RATE_LIMIT_UNTIL

        now = time.time()
        if now < GROQ_RATE_LIMIT_UNTIL:
            wait_seconds = int(GROQ_RATE_LIMIT_UNTIL - now)
            return (
                f"⏳ Groq rate limit reached. Please wait {wait_seconds} seconds before sending a new request.",
                []
            )

        cache_key = hashlib.md5(
            (query + str([c['content'][:50] for c in relevant_chunks[:5]])).encode()
        ).hexdigest()

        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        context, used_chunks = self.fit_context_to_window(relevant_chunks[:12])

        conversation_summary = ""
        if chat_history and len(chat_history) > 1:
            recent = chat_history[-6:]
            conv_lines = []
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg['content'][:200] if len(msg['content']) > 200 else msg['content']
                conv_lines.append(f"{role}: {content}")
            conversation_summary = "\n".join(conv_lines)

        system_prompt = """You are a highly accurate and professional assistant for the Master Biomedical Engineering (MBE) program at Hochschule Anhalt.

CRITICAL RULES:
- Answer EXCLUSIVELY based on the provided document sources or previous conversation history
- If the question is a follow-up, use the conversation history FIRST
- If no relevant information exists: Reply exactly "No sufficient information in the available documents."
- Use the SAME language as the user's question (English, German, or Arabic)
- Be concise, clear, and professional. Use bullet points or numbering when listing items
- Always cite sources briefly (e.g., "According to SPO MBE 2024, page X...")
- NEVER hallucinate, explain your reasoning, or add external knowledge
- For summarization requests of entire documents: Provide a high-level overview including program duration, total credits, main modules/specializations, semester structure, and key regulations
- Always use bullet points or numbered lists for summaries
- Cite multiple pages/sources where possible
- When asked about Master's thesis registration or regulations, prioritize information from "94_B14_SPO_MBE..." or "Notes_on_final_theses..." documents
- For module handbook summaries, list key modules, their credits, and semester distribution if available
- For counting or lists: Be precise and complete
- If a catalog contains both an outline and a detailed list of modules, always prefer the detailed module list when the user asks for module names"""

        data = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
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

            answer = response.json()["choices"][0]["message"]["content"].strip()
            result = (answer, used_chunks)

            if len(self.query_cache) < self.max_cache_size:
                self.query_cache[cache_key] = result

            return result

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After")
                wait_time = int(retry_after) if retry_after else 60

                GROQ_RATE_LIMIT_UNTIL = time.time() + wait_time

                return (
                    f"⛔ Groq rate limit reached.\n"
                    f"⏳ Please wait {wait_time} seconds before trying again.",
                    []
                )

            return f"❌ HTTP Error: {str(e)}", []

        except Exception as e:
            return f"❌ Error: {str(e)}", []


def detect_language(text):
    if LANGDETECT_AVAILABLE:
        try:
            lang = detect(text)
            if lang == 'de':
                return 'de'
            elif lang == 'ar':
                return 'ar'
            else:
                return 'en'
        except:
            pass
    
    import re
    text_lower = text.lower()
    if re.search(r'[äöüß]', text_lower):
        return "de"
    if re.search(r'[ء-ي]', text_lower):
        return "ar"
    return "en"


def translate_query(query, source_lang, target_lang):
    if source_lang == target_lang:
        return query

    if TRANSLATOR_AVAILABLE:
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            return translator.translate(query)
        except:
            pass

    prompt = f"""Translate the following question from {source_lang} to {target_lang}.
Keep it accurate, literal, and academic.
Do NOT explain.

Question:
{query}
"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0
            },
            timeout=30
        )
        return response.json()["choices"][0]["message"]["content"].strip()
    except:
        return query


def get_available_languages(collection):
    langs = set()
    try:
        metas = collection.get(include=["metadatas"])["metadatas"]
        for m in metas:
            if m and "lang" in m:
                langs.add(m["lang"])
    except:
        pass
    return list(langs) if langs else ['en']


def expand_query_multilingual(query, collection):
    user_lang = detect_language(query)
    doc_langs = get_available_languages(collection)

    if not doc_langs or user_lang in doc_langs:
        return [query]

    expanded_queries = [query]
    for lang in doc_langs[:2]:
        if lang != user_lang:
            translated = translate_query(query, user_lang, lang)
            if translated != query:
                expanded_queries.append(translated)

    return expanded_queries


def answer_question_with_groq(query, relevant_chunks, chat_history=None):
    engine = ImprovedChatEngine()
    return engine.answer_question_with_groq(query, relevant_chunks, chat_history)
