#!/usr/bin/env python3
"""
API Wrapper for Kerala Farming RAG Chatbot

This script provides a simple interface for the Node.js backend
to interact with the Python RAG chatbot system.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from chatbot.chatbot import KeralaFarmingChatbot

# Load environment variables
load_dotenv()

class ChatbotAPIWrapper:
    """
    API wrapper for the Kerala Farming Chatbot
    """
    
    def __init__(self):
        """Initialize the chatbot"""
        self.chatbot = None
        self.initialize_chatbot()
    
    def initialize_chatbot(self):
        """Initialize the chatbot with default settings"""
        try:
            # Get configuration from environment or use defaults
            db_path = os.getenv("CHROMA_DB_PATH", "chroma_db")
            llm_type = os.getenv("LLM_TYPE", "openai")
            llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
            top_k = int(os.getenv("TOP_K_DOCUMENTS", "5"))
            
            self.chatbot = KeralaFarmingChatbot(
                db_path=db_path,
                llm_type=llm_type,
                model_name=llm_model,
                temperature=temperature,
                top_k_docs=top_k
            )
            return {"status": "success", "message": "Chatbot initialized successfully"}
        except Exception as e:
            return {"status": "error", "message": f"Failed to initialize chatbot: {str(e)}"}
    
    def ask_question(self, question, include_sources=True):
        """
        Ask a question to the chatbot
        
        Args:
            question (str): User's question
            include_sources (bool): Whether to include source documents
            
        Returns:
            dict: Response with answer and sources
        """
        try:
            if not self.chatbot:
                return {"status": "error", "message": "Chatbot not initialized"}
            
            # Get response from chatbot
            response = self.chatbot.ask(question, include_sources=include_sources)
            
            return {
                "status": "success",
                "question": response["question"],
                "answer": response["answer"],
                "sources": response.get("sources", [])
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing question: {str(e)}",
                "question": question,
                "answer": "I apologize, but I encountered an error while processing your question.",
                "sources": []
            }
    
    def get_health_status(self):
        """Get the health status of the chatbot"""
        try:
            if self.chatbot:
                # Test with a simple query
                test_response = self.chatbot.ask("test", include_sources=False)
                return {
                    "status": "healthy",
                    "message": "Chatbot is running and responsive",
                    "db_path": self.chatbot.db_path,
                    "llm_type": self.chatbot.llm_type,
                    "model": self.chatbot.model_name
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": "Chatbot not initialized"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}"
            }

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Kerala Farming Chatbot API Wrapper")
    parser.add_argument("action", choices=["ask", "health", "init"], 
                       help="Action to perform")
    parser.add_argument("--question", type=str, 
                       help="Question to ask (for 'ask' action)")
    parser.add_argument("--no-sources", action="store_true",
                       help="Don't include source documents in response")
    
    args = parser.parse_args()
    
    # Initialize wrapper
    wrapper = ChatbotAPIWrapper()
    
    if args.action == "init":
        result = wrapper.initialize_chatbot()
    elif args.action == "health":
        result = wrapper.get_health_status()
    elif args.action == "ask":
        if not args.question:
            result = {"status": "error", "message": "Question is required for 'ask' action"}
        else:
            result = wrapper.ask_question(args.question, include_sources=not args.no_sources)
    else:
        result = {"status": "error", "message": "Invalid action"}
    
    # Output JSON result
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
