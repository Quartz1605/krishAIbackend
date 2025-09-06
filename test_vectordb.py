#!/usr/bin/env python3
"""
Quick test to verify vector database functionality
"""
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from embeddings.build_vector_db import VectorDBBuilder

def test_vector_search():
    print("🔍 Testing vector database search...")
    
    try:
        # Initialize builder
        builder = VectorDBBuilder()
        
        # Test search
        test_queries = [
            "rice cultivation in Kerala",
            "pest control methods",
            "soil management",
            "coconut farming"
        ]
        
        for query in test_queries:
            print(f"\n📋 Query: '{query}'")
            docs = builder.search_similar_documents(query, k=2)
            
            if docs:
                print(f"✅ Found {len(docs)} relevant documents:")
                for i, doc in enumerate(docs, 1):
                    filename = doc.metadata.get("filename", "Unknown")
                    preview = doc.page_content[:150].replace('\n', ' ').strip()
                    print(f"   {i}. {filename}")
                    print(f"      Preview: {preview}...")
            else:
                print("❌ No documents found")
        
        print(f"\n✅ Vector database test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_vector_search()
    if success:
        print("\n🎉 Your RAG system is ready!")
        print("Next steps:")
        print("1. Add your OpenAI API key to .env file, or")
        print("2. Install Ollama for local models")
        print("3. Run: python main.py --chat")
    else:
        print("\n❌ There was an issue with the vector database")
