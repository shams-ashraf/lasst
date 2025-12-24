"""
Chat Engine Module - مسؤول عن التواصل مع API والإجابة على الأسئلة
"""
import requests
import uuid
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional

class Chat:
    def __init__(self, config):
        self.config = config
        self.groq_api_key = config['groq_api_key']
        self.groq_model = config['groq_model']
        self.collection = None
        self.client = None
    
    def get_embedding_function(self):
        """Get sentence transformer embedding function"""
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="intfloat/multilingual-e5-large"
        )
    
    def initialize_collection(self, all_chunks: List[str], 
                             all_metadata: List[Dict]) -> bool:
        """Initialize ChromaDB collection with documents"""
        try:
            self.client = chromadb.Client()
            collection_name = f"docs_{uuid.uuid4().hex[:8]}"
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.get_embedding_function()
            )
            
            # Add documents in batches
            batch_size = 500
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i+batch_size]
                metadata_batch = all_metadata[i:i+batch_size]
                self.collection.add(
                    documents=batch,
                    ids=[f"chunk_{i+j}" for j in range(len(batch))],
                    metadatas=metadata_batch
                )
            
            return True
        except Exception as e:
            print(f"Error initializing collection: {str(e)}")
            return False
    
    def search_documents(self, query: str, n_results: int = 10) -> List[Dict]:
        """Search for relevant document chunks"""
        if not self.collection:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Build chunk objects with metadata
            relevant_chunks = []
            for content, metadata in zip(results["documents"][0], 
                                        results["metadatas"][0]):
                relevant_chunks.append({
                    'content': content,
                    'metadata': metadata
                })
            
            return relevant_chunks
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []
    
    def build_context_with_citations(self, relevant_chunks: List[Dict]) -> str:
        """Build context string with proper citations"""
        context_parts = []
        
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
            
            context_parts.append(f"{citation}\n{content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def build_conversation_summary(self, chat_history: List[Dict]) -> str:
        """Build conversation summary for context"""
        if not chat_history or len(chat_history) == 0:
            return "No previous conversation"
        
        recent = chat_history[-6:]  # Last 3 Q&A pairs
        conv_lines = []
        
        for msg in recent:
            role = "User" if msg['role'] == 'user' else "Assistant"
            content_preview = msg['content'][:300]
            conv_lines.append(f"{role}: {content_preview}")
        
        return "\n".join(conv_lines)
    
    def generate_answer(self, query: str, relevant_chunks: List[Dict], 
                       chat_history: Optional[List[Dict]] = None) -> str:
        """Generate answer using Groq API"""
        if not self.groq_api_key:
            return "❌ GROQ_API_KEY not configured"
        
        # Build context and conversation summary
        context = self.build_context_with_citations(relevant_chunks)
        conversation_summary = self.build_conversation_summary(chat_history)
        
        # Prepare API request
        data = {
            "model": self.groq_model,
            "messages": [
                {
                    "role": "system",
                    "content": """You are a precise MBE Document Assistant at Hochschule Anhalt specializing in Biomedical Engineering regulations.

CRITICAL RULES:
1. Answer ONLY from provided sources OR previous conversation if it's a follow-up question
2. ALWAYS cite sources: [Source X, Page Y] or [Source X, Page Y, Table Z]
3. For follow-up questions like "summarize", "tell me more", "explain that", or "what about that":
   - Check the conversation history FIRST
   - Summarize or expand on your PREVIOUS answer
   - Don't search for new information if the question refers to what you just said
4. If user says "summarize that" or "summarize it": Condense your LAST answer (from conversation history)
5. If no relevant info in sources OR history: "No sufficient information in the available documents"
6. Use the SAME language as the question (English/German/Arabic)
7. Be CONCISE - short, direct answers unless asked to elaborate
8. For counting questions: Count precisely and list all items with citations

FOLLOW-UP DETECTION:
- "that", "it", "this", "summarize", "tell me more", "elaborate", "explain further" → Use conversation history
- New factual questions → Use sources

Remember: You're helping MBE students understand their program requirements clearly and accurately."""
                },
                {
                    "role": "user",
                    "content": f"""CONVERSATION HISTORY (use for follow-up questions):
{conversation_summary}

DOCUMENT SOURCES (use for new factual questions):
{context}

CURRENT QUESTION: {query}

Instructions: 
- If this is a follow-up (summarize/elaborate/that/it), answer from conversation history
- If this is a new question, answer from sources with citations
- Always be precise and cite your sources

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
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json=data,
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def answer_question(self, query: str, 
                       chat_history: Optional[List[Dict]] = None) -> tuple:
        """Main method to answer questions
        Returns: (answer, relevant_chunks)
        """
        # Search for relevant documents
        relevant_chunks = self.search_documents(query)
        
        # Generate answer
        answer = self.generate_answer(query, relevant_chunks, chat_history)
        
        return answer, relevant_chunks
