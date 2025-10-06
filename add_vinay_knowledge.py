#!/usr/bin/env python3
"""
Add Vinay's Knowledge to RAG Database
Run this to populate the knowledge base with information about Vinay and Flashback Labs
"""

import requests
import json

# Knowledge base for Vinay Thadem and Flashback Labs
KNOWLEDGE_BASE = [
    {
        "text": "Vinay Thadem is the Co-Founder of Flashback Labs, a cutting-edge AI company specializing in photorealistic digital avatars and real-time interactive systems.",
        "category": "bio"
    },
    {
        "text": "Flashback Labs was founded with the mission to revolutionize human-AI interaction through lifelike digital avatars that can engage in natural conversations.",
        "category": "company"
    },
    {
        "text": "Our core technology stack includes MuseTalk for real-time lip synchronization, 3D Gaussian Splatting for photorealistic rendering, and advanced LLMs with RAG for intelligent conversations.",
        "category": "technology"
    },
    {
        "text": "Flashback Labs' avatar technology is used in customer service, virtual assistants, education, and entertainment applications.",
        "category": "applications"
    },
    {
        "text": "We use RAG (Retrieval-Augmented Generation) to give our avatars access to vast knowledge bases, ensuring accurate and contextual responses.",
        "category": "technology"
    },
    {
        "text": "Vinay Thadem has expertise in AI, machine learning, computer vision, and real-time rendering technologies.",
        "category": "bio"
    },
    {
        "text": "Flashback Labs is committed to ethical AI development, ensuring transparency, privacy, and responsible use of avatar technology.",
        "category": "values"
    },
    {
        "text": "Our avatar platform supports multiple languages and can be customized for different personas and use cases.",
        "category": "features"
    },
    {
        "text": "Vinay believes that the future of AI interaction is conversational, natural, and human-like, which is why Flashback Labs focuses on creating the most realistic digital humans possible.",
        "category": "vision"
    }
]

def add_knowledge_to_server(server_url="http://localhost:8000"):
    """Add knowledge to running server"""
    print("📚 Adding Vinay's knowledge to RAG database...")
    print("=" * 60)

    added = 0
    failed = 0

    for idx, item in enumerate(KNOWLEDGE_BASE, 1):
        try:
            response = requests.post(
                f"{server_url}/add_knowledge",
                params={
                    "text": item["text"],
                    "category": item["category"]
                },
                timeout=10
            )

            if response.status_code == 200:
                print(f"✅ [{idx}/{len(KNOWLEDGE_BASE)}] Added: {item['category']}")
                added += 1
            else:
                print(f"❌ [{idx}/{len(KNOWLEDGE_BASE)}] Failed: {response.status_code}")
                failed += 1

        except Exception as e:
            print(f"❌ [{idx}/{len(KNOWLEDGE_BASE)}] Error: {e}")
            failed += 1

    print("=" * 60)
    print(f"✅ Added: {added} | ❌ Failed: {failed}")

    if failed > 0:
        print("\n⚠️  Some knowledge items failed to add.")
        print("Make sure the server is running: pm2 logs flashback-avatar")


if __name__ == "__main__":
    import sys

    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    print(f"🔗 Connecting to: {server_url}")

    # Check if server is running
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Server is running")
            print(f"   RAG: {health.get('rag', False)}")
            print(f"   Knowledge count: {health.get('knowledge_count', 0)}")
            print()

            add_knowledge_to_server(server_url)
        else:
            print("❌ Server not responding correctly")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("\nMake sure the server is running:")
        print("  pm2 start flashback_avatar_rag_voice.py --name flashback-avatar --interpreter python3")
        sys.exit(1)
