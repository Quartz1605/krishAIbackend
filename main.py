#!/usr/bin/env python3
"""
Kerala Farming RAG Chatbot - Main Application

This script provides a complete RAG-based chatbot for Kerala farmers.
It can build the vector database from PDFs and run the interactive chatbot.
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from embeddings.build_vector_db import VectorDBBuilder
from chatbot.chatbot import KeralaFarmingChatbot


def setup_vector_database(kb_dir: str = "data/kb", db_path: str = "chroma_db", force_rebuild: bool = False):
    """
    Set up the vector database from PDF files.
    """
    print("🔧 Setting up vector database...")

    try:
        builder = VectorDBBuilder(db_path=db_path)
        builder.build_or_load_vector_db(
            kb_directory=kb_dir,
            force_rebuild=force_rebuild
        )

        print("✅ Vector database ready!")
        return True

    except Exception as e:
        print(f"❌ Error setting up vector database: {str(e)}")
        return False


def run_chatbot(db_path: str = "chroma_db", 
                model_name: str = None,
                temperature: float = 0.1,
                top_k: int = 5):
    """
    Run the interactive chatbot.
    Chooses Gemini-2.5 if GEMINI_API_KEY is set, otherwise defaults to Llama2.
    """
    # Decide which model to use
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        model_name = "gemini"
        print("✨ Using Gemini-2.5 model via API key")
    else:
        if model_name is None:
            model_name = "llama2"
        print("⚠️ GEMINI_API_KEY not found. Falling back to Llama2")

    print(f"🚀 Starting Kerala Farming Chatbot with model: {model_name}...")

    try:
        chatbot = KeralaFarmingChatbot(
            db_path=db_path,
            model_name=model_name,
            temperature=temperature,
            top_k_docs=top_k
        )

        chatbot.chat_loop()

    except Exception as e:
        print(f"❌ Error running chatbot: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure the vector database exists (run with --setup-db first)")
        print("2. Ensure your model API key / Ollama is properly set up")
        print("3. Ensure all dependencies are installed: pip install -r requirements.txt")


def check_prerequisites():
    """
    Check if all prerequisites are met.
    """
    issues = []

    # Check if knowledge base directory exists
    kb_dir = Path("data/kb")
    if not kb_dir.exists():
        issues.append("❌ Knowledge base directory 'data/kb' not found")
    else:
        pdf_files = list(kb_dir.glob("*.pdf"))
        if not pdf_files:
            issues.append("❌ No PDF files found in 'data/kb' directory")
        else:
            print(f"✅ Found {len(pdf_files)} PDF files in knowledge base")

    # Check if Ollama is installed (basic check)
    if not Path("/usr/local/bin/ollama").exists() and not Path("C:/Program Files/Ollama").exists():
        issues.append("❌ Ollama not found. Install from https://ollama.ai/")
    else:
        print("✅ Ollama installation detected")

    # Check GEMINI_API_KEY
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️ GEMINI_API_KEY not set. Gemini-2.5 model won't be available.")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Kerala Farming RAG Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --setup-db --chat
  python main.py --setup-db
  python main.py --chat
  python main.py --chat --model llama3
  python main.py --setup-db --force-rebuild
        """
    )

    parser.add_argument("--setup-db", action="store_true", help="Setup/build the vector database from PDFs")
    parser.add_argument("--chat", action="store_true", help="Run the interactive chatbot")
    parser.add_argument("--check", action="store_true", help="Check prerequisites and configuration")
    parser.add_argument("--kb-dir", type=str, default="data/kb", help="Path to knowledge base directory")
    parser.add_argument("--db-path", type=str, default="chroma_db", help="Path to vector database")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild vector database")
    parser.add_argument("--model", type=str, default=None, help="Fallback model (default: llama2)")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")

    args = parser.parse_args()

    if not any([args.setup_db, args.chat, args.check]):
        parser.print_help()
        print("\n🌾 Quick start:")
        print("1. python main.py --check")
        print("2. python main.py --setup-db")
        print("3. python main.py --chat")
        return

    if args.check:
        print("🔍 Checking prerequisites...\n")
        issues = check_prerequisites()
        if issues:
            print("\n⚠️ Issues found:")
            for issue in issues:
                print(f"   {issue}")
            return
        else:
            print("\n✅ All prerequisites met!")
            return

    if args.setup_db:
        success = setup_vector_database(
            kb_dir=args.kb_dir,
            db_path=args.db_path,
            force_rebuild=args.force_rebuild
        )
        if not success:
            print("❌ Failed to setup vector database.")
            return

    if args.chat:
        db_path = Path(args.db_path)
        if not db_path.exists():
            print(f"❌ Vector database not found at '{args.db_path}'")
            print("Please run with --setup-db first.")
            return

        run_chatbot(
            db_path=args.db_path,
            model_name=args.model,
            temperature=args.temperature,
            top_k=args.top_k
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Happy farming! 🌱")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)
