#!/usr/bin/env python3
"""
Disease Detection RAG Assistant - Main Application

This script runs end-to-end image classification followed by RAG-based
solution generation, similar to rag_chatbot's CLI.
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from predict_with_rag import PlantDiseaseAnalyzer


def run_analysis(image_path: str, db_path: str = "chroma_db", model_name: str = None,
                 interactive: bool = False, include_sources: bool = True):
    """
    Run analysis on a single image and optionally enter interactive chat.
    """
    try:
        analyzer = PlantDiseaseAnalyzer(db_path=db_path, model_name=model_name)
        if interactive:
            analyzer.interactive_analysis(image_path)
        else:
            result = analyzer.analyze_image(image_path, include_sources=include_sources)
            analyzer.print_analysis_results(result)
    except Exception as e:
        print(f"❌ Error running analysis: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure chroma_db exists in the disease_detection folder")
        print("2. Ensure your GEMINI_API_KEY is set in .env if using Gemini")
        print("3. Ensure all dependencies are installed: pip install -r requirements.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Disease Detection RAG Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --image virus4.JPG
  python main.py --image virus4.JPG --interactive
  python main.py --image virus4.JPG --model gemini
  python main.py --image path/to/img.jpg --db-path chroma_db
        """
    )

    parser.add_argument("--image", type=str, required=True, help="Path to the image for analysis")
    parser.add_argument("--db-path", type=str, default="chroma_db", help="Path to vector database (copied from rag_chatbot)")
    parser.add_argument("--model", type=str, default=None, help="LLM model: 'gemini' or 'llama2'")
    parser.add_argument("--interactive", action="store_true", help="Run interactive chat after initial analysis")
    parser.add_argument("--no-sources", action="store_true", help="Don't include source documents")

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)

    # Resolve db path relative to disease_detection dir
    db_path = Path(args.db_path)
    if not db_path.is_absolute():
        db_path = str((Path(__file__).parent / db_path).resolve())
    else:
        db_path = str(db_path)

    run_analysis(
        image_path=str(image_path),
        db_path=db_path,
        model_name=args.model,
        interactive=args.interactive,
        include_sources=not args.no_sources,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Stay healthy, plants! 🌱")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

