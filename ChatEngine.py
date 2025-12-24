def answer_question_with_groq(query, relevant_chunks, chat_history=None):
    if not GROQ_API_KEY:
        return "❌ Please set GROQ_API_KEY in environment variables"
   
    # Build context with precise citations
    context_parts = []
    for i, chunk_data in enumerate(relevant_chunks[:10], 1):
        content = chunk_data['content']
        meta = chunk_data['metadata']
        
        # Handle string metadata from ChromaDB
        source = meta.get('source', 'Unknown')
        page = meta.get('page', 'N/A')
        is_table = meta.get('is_table', 'False')
        table_num = meta.get('table_number', 'N/A')
        
        citation = f"[Source {i}: {source}, Page {page}"
        if is_table == 'True' or is_table == True:
            citation += f", Table {table_num}"
        citation += "]"
        
        context_parts.append(f"{citation}\n{content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Build conversation history for follow-ups
    conversation_summary = ""
    if chat_history and len(chat_history) > 0:
        recent = chat_history[-6:]  # Last 3 Q&A pairs
        conv_lines = []
        for msg in recent:
            role = "User" if msg['role'] == 'user' else "Assistant"
            content_preview = msg['content'][:300]  # Longer preview
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
2. ALWAYS cite sources.
3. For follow-up questions like "summarize", "tell me more", "explain that", or "what about that":
   - Check the conversation history FIRST
   - Summarize or expand on your PREVIOUS answer
4. If user says "summarize that" or "summarize it": Condense your LAST answer (from conversation history)
5. If no relevant info in sources OR history: "No sufficient information in the available documents"
6. Use the SAME language as the question (English/German/Arabic)
7. Be CONCISE - short, direct answers unless asked to elaborate
8. For counting questions: Count precisely and list all items with citations
9. Do NOT explain your thought process.
10. Answer directly and clearly.
11. Append the relevant source(s) at the END of the answer.

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
- If this is a new question, answer from sources with citations
- Do NOT include your thought process
- Always append the relevant source(s) at the END of your answer

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
