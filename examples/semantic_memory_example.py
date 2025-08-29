"""
Example usage of Semantic Memory System for Gianna AI Assistant.

This example demonstrates how to use the semantic memory system to store
and retrieve conversation interactions with semantic similarity search.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict

from gianna.memory import MemoryConfig, SemanticMemory
from gianna.memory.embeddings import EmbeddingProvider
from gianna.memory.vectorstore import VectorStoreProvider


def basic_memory_example():
    """Basic example of storing and retrieving interactions."""
    print("🧠 Gianna Semantic Memory - Basic Example")
    print("=" * 50)

    # Create memory configuration
    config = MemoryConfig(
        embedding_provider=EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS,
        vectorstore_provider=VectorStoreProvider.IN_MEMORY,
        similarity_threshold=0.7,
        max_search_results=5,
    )

    # Initialize semantic memory
    try:
        memory = SemanticMemory(config)
        print("✅ Semantic memory initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize memory: {e}")
        return

    # Session ID for this example
    session_id = "example_session_001"

    # Store some example interactions
    print("\n📝 Storing example interactions...")

    interactions = [
        {
            "user_input": "What is machine learning?",
            "assistant_response": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.",
            "context": "Educational question about AI concepts",
        },
        {
            "user_input": "How does neural networks work?",
            "assistant_response": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections and activation functions.",
            "context": "Follow-up question about AI implementation",
        },
        {
            "user_input": "Can you help me with Python programming?",
            "assistant_response": "Absolutely! Python is a versatile programming language. What specific aspect of Python would you like help with? I can assist with syntax, libraries, best practices, or specific problems.",
            "context": "Programming assistance request",
        },
        {
            "user_input": "What are the best practices for data preprocessing?",
            "assistant_response": "Data preprocessing best practices include: handling missing values, removing duplicates, normalizing data, encoding categorical variables, feature selection, and splitting data for training/testing.",
            "context": "Data science methodology question",
        },
        {
            "user_input": "How do I create a web API with Flask?",
            "assistant_response": "To create a web API with Flask: 1) Install Flask, 2) Create app instance, 3) Define routes with @app.route decorators, 4) Return JSON responses, 5) Handle different HTTP methods, 6) Run with app.run().",
            "context": "Specific programming implementation question",
        },
    ]

    # Store interactions
    interaction_ids = []
    for i, interaction in enumerate(interactions, 1):
        interaction_id = memory.store_interaction(
            session_id=session_id,
            user_input=interaction["user_input"],
            assistant_response=interaction["assistant_response"],
            context=interaction["context"],
            interaction_type="conversation",
            metadata={
                "example_number": i,
                "topic": (
                    "programming"
                    if "Python" in interaction["user_input"]
                    or "Flask" in interaction["user_input"]
                    else "ai"
                ),
            },
        )

        if interaction_id:
            interaction_ids.append(interaction_id)
            print(f"  ✅ Stored interaction {i}: {interaction['user_input'][:50]}...")
        else:
            print(f"  ❌ Failed to store interaction {i}")

    print(f"\n📊 Successfully stored {len(interaction_ids)} interactions")

    # Demonstrate semantic search
    print("\n🔍 Demonstrating semantic search...")

    search_queries = [
        "artificial intelligence and deep learning",
        "web development with Python frameworks",
        "data processing and analysis techniques",
        "programming help and coding assistance",
    ]

    for query in search_queries:
        print(f"\n🔎 Searching for: '{query}'")

        similar_interactions = memory.search_similar_interactions(
            query=query, session_id=session_id, max_results=3, similarity_threshold=0.5
        )

        if similar_interactions:
            for j, interaction in enumerate(similar_interactions, 1):
                print(f"  {j}. User: {interaction.user_input}")
                print(f"     Response: {interaction.assistant_response[:100]}...")
                print(f"     Type: {interaction.interaction_type}")
                print()
        else:
            print("  No similar interactions found")

    # Get context summary
    print("\n📋 Generating context summary...")
    context_summary = memory.get_context_summary(session_id)
    print(f"Context Summary:\n{context_summary}")

    # Analyze patterns
    print("\n🔍 Analyzing interaction patterns...")
    patterns = memory.detect_patterns(session_id)
    print("Detected patterns:")
    for key, value in patterns.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")

    # Get memory statistics
    print("\n📈 Memory system statistics...")
    stats = memory.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def advanced_memory_example():
    """Advanced example with clustering and filtering."""
    print("\n\n🧠 Gianna Semantic Memory - Advanced Example")
    print("=" * 50)

    # Create memory configuration with clustering enabled
    config = MemoryConfig(
        embedding_provider=EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS,
        vectorstore_provider=VectorStoreProvider.IN_MEMORY,
        similarity_threshold=0.6,
        max_search_results=10,
        enable_clustering=True,
        cluster_similarity_threshold=0.8,
        min_cluster_size=2,
    )

    try:
        memory = SemanticMemory(config)
        print("✅ Advanced semantic memory initialized")
    except Exception as e:
        print(f"❌ Failed to initialize advanced memory: {e}")
        return

    session_id = "advanced_session_001"

    # Store interactions with different types
    print("\n📝 Storing diverse interactions...")

    diverse_interactions = [
        # Programming cluster
        {
            "user_input": "How do I debug Python code?",
            "assistant_response": "You can debug Python code using the built-in debugger (pdb), IDE debuggers, print statements, logging, or testing frameworks like pytest.",
            "type": "programming",
        },
        {
            "user_input": "What's the difference between list and tuple in Python?",
            "assistant_response": "Lists are mutable (can be changed after creation) while tuples are immutable. Lists use [] brackets, tuples use () parentheses.",
            "type": "programming",
        },
        {
            "user_input": "How to handle exceptions in Python?",
            "assistant_response": "Use try-except blocks to handle exceptions. You can catch specific exceptions or use a general except clause. Always consider using finally for cleanup.",
            "type": "programming",
        },
        # AI/ML cluster
        {
            "user_input": "What is supervised learning?",
            "assistant_response": "Supervised learning uses labeled training data to learn a mapping between inputs and outputs. Examples include classification and regression problems.",
            "type": "ai_ml",
        },
        {
            "user_input": "Explain gradient descent algorithm",
            "assistant_response": "Gradient descent is an optimization algorithm that finds the minimum of a function by iteratively moving in the direction of steepest descent (negative gradient).",
            "type": "ai_ml",
        },
        {
            "user_input": "What is overfitting in machine learning?",
            "assistant_response": "Overfitting occurs when a model learns the training data too well, including noise, leading to poor performance on new, unseen data.",
            "type": "ai_ml",
        },
        # General help
        {
            "user_input": "Can you recommend some good books?",
            "assistant_response": "I'd be happy to recommend books! What genre or topic interests you? Fiction, non-fiction, technical books, or something specific?",
            "type": "general",
        },
        {
            "user_input": "What's the weather like?",
            "assistant_response": "I don't have access to real-time weather data. Please check a weather service or app for current conditions in your area.",
            "type": "general",
        },
    ]

    # Store diverse interactions
    for i, interaction in enumerate(diverse_interactions, 1):
        memory.store_interaction(
            session_id=session_id,
            user_input=interaction["user_input"],
            assistant_response=interaction["assistant_response"],
            interaction_type=interaction["type"],
            metadata={"category": interaction["type"], "sequence": i},
        )
        print(
            f"  ✅ Stored {interaction['type']} interaction: {interaction['user_input'][:40]}..."
        )

    # Demonstrate filtered search
    print("\n🔍 Demonstrating filtered searches...")

    # Search only programming interactions
    print("\n🐍 Programming-related search:")
    programming_results = memory.search_similar_interactions(
        query="coding and software development",
        session_id=session_id,
        interaction_type="programming",
        max_results=5,
    )

    for result in programming_results:
        print(f"  • {result.user_input}")

    # Search only AI/ML interactions
    print("\n🤖 AI/ML-related search:")
    ai_results = memory.search_similar_interactions(
        query="machine learning algorithms and models",
        session_id=session_id,
        interaction_type="ai_ml",
        max_results=5,
    )

    for result in ai_results:
        print(f"  • {result.user_input}")

    # Analyze patterns with clustering
    print("\n🔍 Advanced pattern analysis...")
    patterns = memory.detect_patterns(session_id)

    print("Interaction type distribution:")
    for interaction_type, count in patterns.get("interaction_types", {}).items():
        print(f"  {interaction_type}: {count}")

    if "clusters" in patterns:
        print(f"\nDetected {len(patterns['clusters'])} clusters:")
        for cluster_id, interaction_ids in patterns["clusters"].items():
            print(f"  Cluster {cluster_id[:8]}...: {len(interaction_ids)} interactions")

    # Context summary
    print("\n📋 Advanced context summary...")
    summary = memory.get_context_summary(session_id, max_interactions=8)
    print(summary)


def fallback_example():
    """Example demonstrating fallback to simpler providers when advanced ones aren't available."""
    print("\n\n🧠 Gianna Semantic Memory - Fallback Example")
    print("=" * 50)

    # Try different configurations in order of preference
    configs_to_try = [
        {
            "name": "OpenAI + ChromaDB",
            "config": MemoryConfig(
                embedding_provider=EmbeddingProvider.OPENAI,
                vectorstore_provider=VectorStoreProvider.CHROMA,
            ),
        },
        {
            "name": "SentenceTransformers + FAISS",
            "config": MemoryConfig(
                embedding_provider=EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS,
                vectorstore_provider=VectorStoreProvider.FAISS,
            ),
        },
        {
            "name": "HuggingFace + In-Memory",
            "config": MemoryConfig(
                embedding_provider=EmbeddingProvider.LOCAL_HUGGINGFACE,
                vectorstore_provider=VectorStoreProvider.IN_MEMORY,
            ),
        },
    ]

    memory = None
    working_config = None

    for config_info in configs_to_try:
        print(f"\n🔧 Trying {config_info['name']}...")
        try:
            memory = SemanticMemory(config_info["config"])
            working_config = config_info["name"]
            print(f"✅ Successfully initialized with {config_info['name']}")
            break
        except Exception as e:
            print(f"❌ Failed to initialize {config_info['name']}: {e}")

    if not memory:
        print("❌ No memory configuration worked")
        return

    # Quick test with the working configuration
    session_id = "fallback_test_session"

    memory.store_interaction(
        session_id=session_id,
        user_input="Hello, this is a test interaction",
        assistant_response="Hello! This is a test response to verify the memory system is working.",
        context="System test",
    )

    # Search for the stored interaction
    results = memory.search_similar_interactions(
        query="test interaction hello", session_id=session_id
    )

    if results:
        print(f"\n✅ Memory system working with {working_config}")
        print(f"Found {len(results)} matching interactions")
    else:
        print(f"\n⚠️ Memory system initialized but search didn't find results")

    # Show final stats
    stats = memory.get_memory_stats()
    print(f"\nFinal configuration: {working_config}")
    print(f"Embedding provider: {stats.get('embedding_provider')}")
    print(f"Vector store provider: {stats.get('vector_store_provider')}")


if __name__ == "__main__":
    try:
        # Run basic example
        basic_memory_example()

        # Run advanced example
        advanced_memory_example()

        # Run fallback example
        fallback_example()

        print("\n🎉 All semantic memory examples completed!")

    except KeyboardInterrupt:
        print("\n\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()
