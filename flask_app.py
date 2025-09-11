#!/usr/bin/env python3
"""
Flask Backend for Kerala Farming Chatbot
This module provides REST API endpoints to interact with the KeralaFarmingChatbot.
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Add the chatbot module path
sys.path.append(str(Path(__file__).parent / "rag_chatbot" / "src"))

# Import the chatbot
try:
    from rag_chatbot.src.chatbot.chatbot import KeralaFarmingChatbot
except ImportError as e:
    print(f"Error importing KeralaFarmingChatbot: {e}")
    print("Make sure the chatbot module is accessible and all dependencies are installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Global chatbot instance
chatbot = None

def initialize_chatbot():
    """Initialize the chatbot instance with error handling."""
    global chatbot
    try:
        # Use the chroma_db path relative to the rag_chatbot directory
        db_path = str(Path(__file__).parent / "rag_chatbot" / "chroma_db")
        
        logger.info("Initializing Kerala Farming Chatbot...")
        chatbot = KeralaFarmingChatbot(
            db_path=db_path,
            model_name=None,  # Auto-detect model
            temperature=0.1,
            top_k_docs=5
        )
        logger.info("Chatbot initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({
        "status": "ok",
        "message": "Kerala Farming Chatbot API is running",
        "chatbot_ready": chatbot is not None
    }), 200

@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint to interact with the Kerala Farming Chatbot.
    
    Expected JSON payload:
    {
        "query": "Your farming question here"
    }
    
    Returns:
    {
        "success": true,
        "question": "Your farming question here",
        "answer": "Chatbot response",
        "sources": [...]
    }
    """
    try:
        # Check if chatbot is initialized
        if chatbot is None:
            return jsonify({
                "success": False,
                "error": "Chatbot is not initialized. Please check server logs."
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided. Please send a JSON payload."
            }), 400
        
        # Extract query from the payload
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({
                "success": False,
                "error": "Missing or empty 'query' field in JSON payload."
            }), 400
        
        # Get response from chatbot
        logger.info(f"Processing query: {query[:50]}...")
        response = chatbot.ask(query, include_sources=True)
        
        # Format and return response
        return jsonify({
            "success": True,
            "question": response.get("question", query),
            "answer": response.get("answer", ""),
            "sources": response.get("sources", [])
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"An error occurred while processing your request: {str(e)}"
        }), 500

@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API information."""
    return jsonify({
        "message": "Kerala Farming Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health - Check API health",
            "chat": "POST /chat - Send farming queries",
            "home": "GET / - This endpoint"
        },
        "usage": {
            "chat_endpoint": {
                "method": "POST",
                "url": "/chat",
                "payload": {
                    "query": "Your farming question here"
                },
                "example": {
                    "query": "What are the best crops to grow in Kerala during monsoon season?"
                }
            }
        }
    }), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/health", "/chat"]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        "success": False,
        "error": "Method not allowed for this endpoint",
        "tip": "Use POST for /chat, GET for /health and /"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "message": "Please check server logs for details"
    }), 500

def main():
    """Main function to run the Flask application."""
    print("=" * 60)
    print("🌾 Kerala Farming Chatbot API Server 🌾")
    print("=" * 60)
    
    # Initialize chatbot
    if not initialize_chatbot():
        print("\n❌ Failed to initialize chatbot. Please check:")
        print("1. The vector database exists (run build_vector_db.py first)")
        print("2. Gemini API key is set in .env or Ollama is accessible")
        print("3. All required packages are installed")
        sys.exit(1)
    
    print(f"\n🚀 Starting Flask server...")
    print(f"📍 Server will be available at: http://127.0.0.1:8000")
    print(f"📋 API Endpoints:")
    print(f"   • GET  /health - Health check")
    print(f"   • POST /chat   - Send farming queries")
    print(f"   • GET  /       - API documentation")
    print(f"\n💡 Example usage:")
    print(f'   curl -X POST http://127.0.0.1:8000/chat \\')
    print(f'        -H "Content-Type: application/json" \\')
    print(f'        -d \'{{\"query\": \"What crops grow well in Kerala?\"}}\'')
    print(f"\n🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Run Flask app
        app.run(
            host='127.0.0.1',
            port=8000,
            debug=True,
            threaded=True
        )
    except KeyboardInterrupt:
        print(f"\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")

if __name__ == '__main__':
    main()
