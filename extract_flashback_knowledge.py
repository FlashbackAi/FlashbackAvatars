"""
Extract Flashback Labs knowledge from NotionDoc.md and add to RAG database.
"""

import chromadb
from chromadb.utils import embedding_functions
import uuid

# Knowledge extracted from NotionDoc.md
FLASHBACK_KNOWLEDGE = [
    # TEEPIN Platform
    {
        "text": "TEEPIN is a decentralized framework for training and deploying Private AI Avatars. It guarantees ownership through wallet-tied decentralized storage, protects privacy with Trusted Execution Environments (TEEs) for confidential compute, and enforces control with on-chain programmable permissions.",
        "category": "platform",
        "source": "NotionDoc"
    },
    {
        "text": "TEEPIN uses decentralized storage with Shelby, 0G, and BNB Greenfield. Storage buckets provide access logs and programmable permissions, ensuring users maintain full control over their data.",
        "category": "technology",
        "source": "NotionDoc"
    },
    {
        "text": "TEEPIN implements confidential compute using Trusted Execution Environments (TEEs) with Oasis ROFL. TEEs provide hardware-level memory isolation, sealing keys, and attestation proofs that computations are untampered. Raw data is never visible outside enclaves.",
        "category": "security",
        "source": "NotionDoc"
    },

    # Flashback 1.0: Private AI Memories
    {
        "text": "Flashback 1.0 is a Private AI Memories platform that builds the foundation dataset for avatar training. It includes a recognition layer that detects faces, identifies speaker embeddings and emotions, and classifies scenes like weddings or birthdays.",
        "category": "product",
        "source": "NotionDoc"
    },
    {
        "text": "Flashback 1.0 uses Person Cards to organize relationships. Each card includes name, relationship, gender, and verified identity. All photos, videos, and voice clips associated with a person are linked to their card.",
        "category": "features",
        "source": "NotionDoc"
    },
    {
        "text": "Flashback 1.0 collects memories as real life stories through narrated voice notes, written stories, guided quizzes that elicit structured details like favorite foods and songs, and conversations with Meemaw the voice assistant. These inputs are parsed into structured metadata including timelines and attributes.",
        "category": "features",
        "source": "NotionDoc"
    },
    {
        "text": "Flashback 1.0's generation layer can synthesize new photos consistent with existing media, reconstruct short video clips for missing moments, reconstruct voices using speaker embeddings, and animate static images into 3-second Live Photos with blinking, smiling, and hugging.",
        "category": "features",
        "source": "NotionDoc"
    },

    # Flashback 2.0: Private AI Avatars
    {
        "text": "Flashback 2.0 creates Private AI Avatars that are lifelike, contextual, and interactive. It uses multimodal inputs including visual likeness from photos and videos, motion and micro-expressions from video analysis, voiceprints from audio samples, and personality embeddings derived from real life stories.",
        "category": "product",
        "source": "NotionDoc"
    },
    {
        "text": "Flashback 2.0 features a Hallucinator Engine that expands limited datasets by imagining perspectives not captured. For example, given a photo of a father holding a child, it generates the same scene from the child's perspective, providing avatars with memory expansions that enrich context.",
        "category": "features",
        "source": "NotionDoc"
    },
    {
        "text": "Flashback 2.0 avatars combine visual models that learn face and expression embeddings, voice models that synthesize natural cadence and emotion, personality models anchored to quiz data and stories, and a unified avatar model that creates interactive identities.",
        "category": "technology",
        "source": "NotionDoc"
    },
    {
        "text": "Flashback 2.0 produces conversational avatars accessible by text or voice, media generation that creates new images and videos featuring the avatar, and memory recreation where avatars can replay or reconstruct user-defined stories enriched with imagined perspectives.",
        "category": "features",
        "source": "NotionDoc"
    },

    # Business Model & Tokenomics
    {
        "text": "The $TEE token is used to access compute for training avatars and inference for media generation, earn from usage by participating in brand campaigns or contributing rare datasets, stake for governance and priority compute lanes. Burn mechanisms ensure token scarcity for training-heavy actions and campaign fees.",
        "category": "tokenomics",
        "source": "NotionDoc"
    },
    {
        "text": "Flashback Labs has multiple revenue streams: $20 per month premium subscription with unlimited high-resolution generations, UGC campaigns where brands pay users in $TEE tokens, PaaS white-label licensing for enterprises, compute-as-a-service with token-metered billing, and community contributions rewarding users who provide rare or unique memories.",
        "category": "business",
        "source": "NotionDoc"
    },

    # Market & Vision
    {
        "text": "Flashback Labs targets three user segments: Explorers who are AI enthusiasts experimenting casually, Nostalgics motivated by personal stories and family preservation, and Legacy Keepers preserving memories of lost loved ones or heritage with high willingness to pay.",
        "category": "market",
        "source": "NotionDoc"
    },
    {
        "text": "The market opportunity includes generative AI photo and video at $68B+, AI training dataset market at $18B+, decentralized AI at $46B+, and UGC marketing at $130B+.",
        "category": "market",
        "source": "NotionDoc"
    },

    # Roadmap
    {
        "text": "Flashback's roadmap includes Q3 2025 beta launch with recognition and tagging on Android, iOS, and Solana Mobile, Q4 2025 Hallucinator engine enabling alternate perspectives, Q1 2026 personalized avatars trained on real life stories and multimodal data, and Q2 2026 voice likeness module with staking-based compute prioritization.",
        "category": "roadmap",
        "source": "NotionDoc"
    },

    # Privacy & Security
    {
        "text": "Unlike centralized AI platforms that control user data, TEEPIN ensures confidential data residency where personal stories remain encrypted in wallet-controlled storage, TEE attestation provides hardware-level proofs that training and inference run inside secure enclaves, consent enforcement is immutable and logged on-chain, and programmable permissions let users specify who can read, generate, or train with their memories.",
        "category": "security",
        "source": "NotionDoc"
    },

    # Applications
    {
        "text": "Flashback has applications in personal use for creating avatars that recall family stories, preserve cultural legacies, and animate old photos, commercial use for user-generated campaigns, reviews, and product promotions, and enterprise white-label deployments for cultural archiving, memorials, and education.",
        "category": "applications",
        "source": "NotionDoc"
    },

    # Team & Mission
    {
        "text": "Vinay Thadem is the Co-Founder of Flashback Labs. The mission is to deliver privacy-first AI for personal memories. Unlike generic AI clones, Flashback avatars are trained on photos, videos, voices, and most importantly real life stories told directly by users, with all computation protected by TEEs and governed by on-chain consent.",
        "category": "bio",
        "source": "NotionDoc"
    },
]


def add_flashback_knowledge():
    """Add Flashback knowledge to RAG database."""
    print("🔧 Initializing RAG database...")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="./rag_db")

    # Create embedding function
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Get or create collection
    try:
        collection = client.get_collection(
            name="vinay_knowledge",
            embedding_function=embedding_fn
        )
        print("✅ Using existing collection 'vinay_knowledge'")
    except:
        collection = client.create_collection(
            name="vinay_knowledge",
            embedding_function=embedding_fn,
            metadata={"description": "Vinay Thadem and Flashback Labs knowledge base"}
        )
        print("✅ Created new collection 'vinay_knowledge'")

    # Add knowledge items
    print(f"\n📝 Adding {len(FLASHBACK_KNOWLEDGE)} Flashback knowledge items...")

    ids = []
    documents = []
    metadatas = []

    for item in FLASHBACK_KNOWLEDGE:
        doc_id = str(uuid.uuid4())
        ids.append(doc_id)
        documents.append(item["text"])
        metadatas.append({
            "category": item["category"],
            "source": item["source"]
        })

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print(f"✅ Added {len(FLASHBACK_KNOWLEDGE)} knowledge items to RAG")

    # Test retrieval
    print("\n🔍 Testing knowledge retrieval...")
    test_queries = [
        "What is TEEPIN?",
        "Tell me about Flashback 1.0",
        "What is the Hallucinator Engine?",
        "What is the $TEE token used for?",
    ]

    for query in test_queries:
        results = collection.query(
            query_texts=[query],
            n_results=1
        )
        if results['documents']:
            print(f"\n❓ Query: {query}")
            print(f"✅ Result: {results['documents'][0][0][:200]}...")

    # Show collection stats
    print(f"\n📊 Collection Stats:")
    print(f"   Total documents: {collection.count()}")

    print("\n✅ Flashback knowledge successfully added to RAG!")


if __name__ == "__main__":
    add_flashback_knowledge()
