#!/usr/bin/env python3
"""
Local LLM with RAG for Real-Time Avatar Conversations
Uses ollama for fast local inference with vector search for context
"""

import asyncio
import json
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Vector database and embeddings
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("⚠️  ChromaDB not available. Install with: pip install chromadb")

# Local LLM inference
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Ollama not available. Install with: pip install ollama")

# Alternative: Transformers for direct model loading
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

@dataclass
class ConversationMessage:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float
    context_used: List[str] = None

class LocalRAGDatabase:
    """Vector database for RAG context retrieval"""

    def __init__(self, db_path: str = "rag_database"):
        self.db_path = db_path
        self.client = None
        self.collection = None

        if CHROMA_AVAILABLE:
            self.setup_database()
        else:
            print("⚠️  ChromaDB not available, RAG functionality disabled")

    def setup_database(self):
        """Initialize ChromaDB for vector storage"""
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)

            # Use sentence transformers for embeddings
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )

            self.collection = self.client.get_or_create_collection(
                name="avatar_knowledge",
                embedding_function=sentence_transformer_ef
            )

            print("✅ RAG database initialized")

        except Exception as e:
            print(f"❌ Failed to initialize RAG database: {e}")
            self.client = None
            self.collection = None

    def add_knowledge(self, texts: List[str], metadata: List[Dict] = None):
        """Add knowledge to the vector database"""
        if not self.collection:
            return False

        try:
            ids = [f"doc_{i}" for i in range(len(texts))]
            metadata = metadata or [{"source": f"doc_{i}"} for i in range(len(texts))]

            self.collection.add(
                documents=texts,
                metadatas=metadata,
                ids=ids
            )

            print(f"✅ Added {len(texts)} documents to knowledge base")
            return True

        except Exception as e:
            print(f"❌ Failed to add knowledge: {e}")
            return False

    def search_context(self, query: str, n_results: int = 3) -> List[str]:
        """Search for relevant context based on query"""
        if not self.collection:
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )

            return results['documents'][0] if results['documents'] else []

        except Exception as e:
            print(f"❌ Context search failed: {e}")
            return []

class LocalLLMProcessor:
    """Local LLM processor with multiple backend options"""

    def __init__(self, model_name: str = "llama3.2:3b", backend: str = "ollama"):
        self.model_name = model_name
        self.backend = backend
        self.conversation_history: List[ConversationMessage] = []
        self.rag_db = LocalRAGDatabase()

        # Initialize based on backend
        if backend == "ollama" and OLLAMA_AVAILABLE:
            self.initialize_ollama()
        elif backend == "transformers" and TRANSFORMERS_AVAILABLE:
            self.initialize_transformers()
        else:
            print(f"❌ Backend '{backend}' not available")

    def initialize_ollama(self):
        """Initialize Ollama backend"""
        try:
            # Test if model is available
            models = ollama.list()

            # Handle different response formats
            if isinstance(models, dict) and 'models' in models:
                model_names = [model.get('name', model.get('model', '')) for model in models['models']]
            else:
                model_names = []

            if self.model_name not in model_names:
                print(f"📥 Downloading {self.model_name} model...")
                ollama.pull(self.model_name)

            print(f"✅ Ollama initialized with {self.model_name}")

        except Exception as e:
            print(f"❌ Ollama initialization failed: {e}")
            print(f"   Continuing with fallback LLM...")

    def initialize_transformers(self):
        """Initialize Transformers backend"""
        try:
            model_map = {
                "phi-3.5": "microsoft/Phi-3.5-mini-instruct",
                "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
                "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct"
            }

            hf_model_name = model_map.get(self.model_name, self.model_name)

            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

            print(f"✅ Transformers initialized with {hf_model_name}")

        except Exception as e:
            print(f"❌ Transformers initialization failed: {e}")

    def add_default_knowledge(self):
        """Add default knowledge base for avatar conversations"""
        default_knowledge = [
            "I am Vinay Thadem, Co-founder of Flashback Labs.",
            "Flashback Labs is an innovative AI company focused on cutting-edge technology solutions.",
            "I specialize in AI, machine learning, computer vision, and real-time interactive systems.",
            "I can discuss technology, programming, artificial intelligence, and business topics.",
            "I aim to be helpful, informative, and engaging in conversations about tech and innovation.",
            "I use advanced AI technology to provide real-time conversational experiences.",
            "My responses are generated using local language models for privacy and speed.",
            "I can help with technical questions, AI explanations, and business discussions.",
            "I support real-time conversation with voice input and video responses.",
            "The technology behind this uses EchoMimic v3 for generating video responses with lip-sync.",
            "Flashback Labs develops cutting-edge AI solutions for various industries.",
            "I have extensive experience in AI research, development, and practical applications."
        ]

        metadata = [{"source": "default", "category": "avatar_info"} for _ in default_knowledge]
        self.rag_db.add_knowledge(default_knowledge, metadata)

    async def process_query(self, user_input: str, use_rag: bool = True) -> str:
        """Process user query with optional RAG context"""
        start_time = time.time()

        # Get RAG context if enabled
        context_docs = []
        if use_rag and self.rag_db.collection:
            context_docs = self.rag_db.search_context(user_input, n_results=2)

        # Build prompt with context
        prompt = self.build_prompt(user_input, context_docs)

        # Generate response based on backend
        if self.backend == "ollama" and OLLAMA_AVAILABLE:
            response = await self.generate_ollama(prompt)
        elif self.backend == "transformers" and TRANSFORMERS_AVAILABLE:
            response = await self.generate_transformers(prompt)
        else:
            response = self.fallback_response(user_input)

        # Record conversation
        processing_time = time.time() - start_time

        user_msg = ConversationMessage("user", user_input, start_time)
        assistant_msg = ConversationMessage(
            "assistant", response, time.time(),
            context_used=[doc[:100] for doc in context_docs]
        )

        self.conversation_history.extend([user_msg, assistant_msg])

        print(f"⚡ LLM response generated in {processing_time:.2f}s")

        return response

    def build_prompt(self, user_input: str, context_docs: List[str]) -> str:
        """Build prompt with RAG context and conversation history"""

        # System prompt for avatar personality
        system_prompt = """You are Vinay Thadem, Co-founder of Flashback Labs. You are engaging, knowledgeable, and conversational about technology and AI.
Keep responses concise and natural for spoken conversation (1-3 sentences typically).
You appear as a professional tech entrepreneur in business attire."""

        # Add RAG context if available
        context_section = ""
        if context_docs:
            context_section = "\n\nRelevant context:\n" + "\n".join(f"- {doc}" for doc in context_docs[:2])

        # Recent conversation history (last 4 messages)
        history_section = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-4:]
            history_section = "\n\nRecent conversation:\n"
            for msg in recent_history:
                role = "User" if msg.role == "user" else "Assistant"
                history_section += f"{role}: {msg.content}\n"

        # Build full prompt
        full_prompt = f"""{system_prompt}{context_section}{history_section}

User: {user_input}
Assistant:"""

        return full_prompt

    async def generate_ollama(self, prompt: str) -> str:
        """Generate response using Ollama"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 150,  # Keep responses concise for speech
                    }
                )
            )

            return response['response'].strip()

        except Exception as e:
            print(f"❌ Ollama generation failed: {e}")
            return self.fallback_response("")

    async def generate_transformers(self, prompt: str) -> str:
        """Generate response using Transformers"""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.generate(
                        inputs,
                        max_new_tokens=150,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                )

            response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            print(f"❌ Transformers generation failed: {e}")
            return self.fallback_response("")

    def fallback_response(self, user_input: str) -> str:
        """Fallback response when LLM is unavailable"""
        responses = [
            "I'm processing your request. Could you tell me more about that?",
            "That's interesting! I'd like to help you with that.",
            "Thanks for sharing that with me. What else would you like to know?",
            "I understand. Is there anything specific you'd like me to help with?",
        ]

        return responses[len(user_input) % len(responses)]

# Integration with avatar server
class EnhancedLLMProcessor:
    """Enhanced LLM processor for avatar server integration"""

    def __init__(self):
        self.processor = LocalLLMProcessor(
            model_name="llama3.2:3b",  # Fast 3B model
            backend="ollama"
        )

        # Add default knowledge
        self.processor.add_default_knowledge()

    async def process_avatar_query(self, user_input: str) -> Dict:
        """Process query and return structured response for avatar"""
        response_text = await self.processor.process_query(user_input, use_rag=True)

        # Analyze response for avatar behavior
        emotion = self.detect_emotion(response_text)
        emphasis_words = self.find_emphasis_words(response_text)

        return {
            "text": response_text,
            "emotion": emotion,
            "emphasis_words": emphasis_words,
            "estimated_duration": len(response_text.split()) * 0.4,  # ~400ms per word
            "should_gesture": len(response_text.split()) > 10
        }

    def detect_emotion(self, text: str) -> str:
        """Simple emotion detection for avatar expression"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['excited', 'great', 'amazing', 'wonderful']):
            return 'excited'
        elif any(word in text_lower for word in ['sorry', 'unfortunately', 'problem']):
            return 'concerned'
        elif any(word in text_lower for word in ['think', 'consider', 'perhaps']):
            return 'thoughtful'
        else:
            return 'neutral'

    def find_emphasis_words(self, text: str) -> List[str]:
        """Find words that should be emphasized in speech"""
        emphasis_words = []
        words = text.split()

        for word in words:
            if word.isupper() or word.endswith('!') or word.startswith('*'):
                emphasis_words.append(word.strip('*!.,'))

        return emphasis_words

# Quick setup function
def setup_local_llm():
    """Quick setup for local LLM"""
    print("🧠 Setting up Local LLM with RAG...")

    # Check ollama installation
    try:
        import ollama
        models = ollama.list()
        print(f"✅ Ollama available with {len(models['models'])} models")

        # Recommend fast models
        recommended_models = [
            "llama3.2:3b",     # Fastest, good quality
            "phi3.5:3.8b",     # Balanced speed/quality
            "qwen2.5:7b"       # Best quality, slower
        ]

        print("\n🚀 Recommended models for real-time chat:")
        for model in recommended_models:
            print(f"   • {model}")

        print("\n💡 To install a model: ollama pull llama3.2:3b")

    except ImportError:
        print("❌ Ollama not installed")
        print("📥 Install ollama: https://ollama.ai/")

    return True

if __name__ == "__main__":
    setup_local_llm()