import os
import sys
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI



# Load .env variables
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))
from embeddings.build_vector_db import VectorDBBuilder

# Import LLMs
try:
    from langchain_community.chat_models import ChatOpenAI  # For Gemini via OpenAI-compatible API
except ImportError:
    ChatOpenAI = None

try:
    from langchain_community.chat_models import ChatOllama
except ImportError:
    ChatOllama = None


class KeralaFarmingChatbot:
    """
    RAG-based chatbot for Kerala farming queries using Gemini (preferred) or Ollama.
    """
    
    def __init__(self, 
                 db_path: str = "./chroma_db",
                 model_name: str = None,  # Optional, auto-select if None
                 temperature: float = 0.1,
                 top_k_docs: int = 5):
        self.db_path = db_path
        self.temperature = temperature
        self.top_k_docs = top_k_docs

        # Initialize vector database
        print("Initializing vector database...", file=sys.stderr)
        self.vector_db_builder = VectorDBBuilder(db_path=db_path)
        self.retriever = self.vector_db_builder.get_retriever(k=top_k_docs)

        # Determine which LLM to use
        self.model_name = model_name or self._detect_model()
        print(f"Using LLM: {self.model_name}", file=sys.stderr)
        self.llm = self._initialize_llm()

        # Create custom prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Create QA chain
        self.qa_chain = self._create_qa_chain()
        
        print("Kerala Farming Chatbot initialized successfully!", file=sys.stderr)

    def _detect_model(self) -> str:
        """
        Detect which model to use based on environment.
        Prefer Gemini if API key exists, else Ollama.
        """
        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        return "llama2"

    def _initialize_llm(self):
        """
        Initialize Gemini or Ollama based on model_name.
        """
        if self.model_name == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in .env")
            
            
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",   # or gemini-1.5-pro, gemini-1.5-flash, etc.
                temperature=self.temperature,
                google_api_key=api_key
            )

        elif self.model_name == "llama2":
            if ChatOllama is None:
                raise ImportError("langchain.chat_models.ChatOllama is required for Ollama integration")
            return ChatOllama(
                model="llama2",
                temperature=self.temperature
            )
        else:
            raise ValueError("Unsupported model_name. Use 'gemini' or 'llama2'.")

    def _create_prompt_template(self) -> PromptTemplate:
        template = """You are an AI assistant specialized in Kerala farming and agriculture. 
You help Kerala farmers with their agricultural questions based on the provided context.

Context Information:
{context}

Human Question: {question}

Instructions:
1. Provide accurate, practical advice based on the context provided
2. Focus specifically on Kerala's climate, soil conditions, and farming practices
3. If the question is about crops, mention suitable varieties for Kerala
4. Include information about pest management, soil care, or seasonal considerations when relevant
5. If you cannot find specific information in the context, say so and provide general guidance
6. Keep your response helpful, clear, and practical for farmers
7. Use simple language that farmers can easily understand
8. If applicable, mention any government schemes or support available

Answer:"""
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def _create_qa_chain(self):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.retriever.get_relevant_documents(query)

    def ask(self, question: str, include_sources: bool = True) -> Dict[str, any]:
        try:
            response = self.qa_chain.invoke({"query": question})
            result = {
                "question": question,
                "answer": response["result"],
                "sources": []
            }
            if include_sources and "source_documents" in response:
                sources = []
                for doc in response["source_documents"]:
                    source_info = {
                        "filename": doc.metadata.get("filename", "Unknown"),
                        "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    }
                    sources.append(source_info)
                result["sources"] = sources
            return result
        except Exception as e:
            return {
                "question": question,
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": []
            }

    def chat_loop(self):
        print("\n" + "="*60)
        print("🌾 Kerala Farming Assistant Chatbot 🌾")
        print("Ask me anything about farming in Kerala!")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("="*60 + "\n")
        while True:
            try:
                question = input("\n🧑‍🌾 You: ").strip()
                if question.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n🤖 Assistant: Thank you for using Kerala Farming Assistant! Happy farming! 🌱")
                    break
                if not question:
                    print("\n🤖 Assistant: Please ask a question about farming in Kerala.")
                    continue
                print("\n🤖 Assistant: Let me help you with that...")
                response = self.ask(question)
                print(f"\n{response['answer']}")
                if response['sources']:
                    print("\n📚 Sources:")
                    for i, source in enumerate(response['sources'], 1):
                        print(f"   {i}. {source['filename']}")
            except KeyboardInterrupt:
                print("\n\n🤖 Assistant: Thank you for using Kerala Farming Assistant! Happy farming! 🌱")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try asking your question again.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Kerala Farming RAG Chatbot")
    parser.add_argument("--db_path", type=str, default="./chroma_db")
    parser.add_argument("--model", type=str, default=None, help="Optional: 'gemini' or 'llama2'")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=5)
    
    args = parser.parse_args()
    
    try:
        chatbot = KeralaFarmingChatbot(
            db_path=args.db_path,
            model_name=args.model,
            temperature=args.temperature,
            top_k_docs=args.top_k
        )
        chatbot.chat_loop()
    except Exception as e:
        print(f"❌ Error initializing chatbot: {str(e)}")
        print("\nMake sure:")
        print("1. The vector database exists (run build_vector_db.py first)")
        print("2. Gemini API key is set in .env or Ollama is accessible")
        print("3. All required packages are installed")
        sys.exit(1)


if __name__ == "__main__":
    main()



