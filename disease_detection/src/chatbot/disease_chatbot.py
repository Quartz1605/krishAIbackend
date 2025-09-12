import os
import sys
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load .env variables
load_dotenv()

# Import LLMs
try:
    from langchain_community.chat_models import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_community.chat_models import ChatOllama
except ImportError:
    ChatOllama = None


class DiseaseDetectionChatbot:
    """
    RAG-based chatbot for plant disease solutions using Gemini (preferred) or Ollama.
    Takes disease predictions and generates treatment/prevention advice.
    """
    
    def __init__(self, 
                 db_path: str = "./chroma_db",
                 model_name: str = None,
                 temperature: float = 0.1,
                 top_k_docs: int = 5):
        self.db_path = db_path
        self.temperature = temperature
        self.top_k_docs = top_k_docs

        # Initialize embeddings (same as original system)
        print("Initializing embeddings...", file=sys.stderr)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize vector database
        print("Loading vector database...", file=sys.stderr)
        self.vectorstore = self._load_vector_store()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k_docs})

        # Determine which LLM to use
        self.model_name = model_name or self._detect_model()
        print(f"Using LLM: {self.model_name}", file=sys.stderr)
        self.llm = self._initialize_llm()

        # Create custom prompt templates
        self.disease_prompt = self._create_disease_prompt_template()
        self.prevention_prompt = self._create_prevention_prompt_template()
        
        # Create QA chains
        self.disease_qa_chain = self._create_qa_chain(self.disease_prompt)
        self.prevention_qa_chain = self._create_qa_chain(self.prevention_prompt)
        
        print("Disease Detection Chatbot initialized successfully!", file=sys.stderr)

    def _load_vector_store(self):
        """Load existing ChromaDB vector store."""
        try:
            if Path(self.db_path).exists():
                vectorstore = Chroma(
                    persist_directory=str(self.db_path),
                    embedding_function=self.embeddings,
                    collection_name="kerala_farming_kb"
                )
                return vectorstore
            else:
                raise FileNotFoundError(f"Vector database not found at {self.db_path}")
        except Exception as e:
            raise Exception(f"Error loading vector store: {str(e)}")

    def _detect_model(self) -> str:
        """Detect which model to use based on environment."""
        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        return "llama2"

    def _initialize_llm(self):
        """Initialize Gemini or Ollama based on model_name."""
        if self.model_name == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in .env")
            
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
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

    def _create_disease_prompt_template(self) -> PromptTemplate:
        """Create prompt template for disease treatment advice."""
        template = """You are an expert plant pathologist and agricultural advisor. 
A plant has been diagnosed with the following condition: {disease_name}

Based on the provided agricultural knowledge base, provide comprehensive treatment and management advice.

Context Information:
{context}

Disease Detected: {disease_name}
Plant Type: {plant_type}

Instructions:
1. Provide immediate treatment steps for this specific disease
2. Recommend appropriate fungicides, pesticides, or organic treatments
3. Suggest cultural practices to manage the disease
4. Include timing and application instructions
5. Mention any precautions or safety measures
6. Provide information about disease progression if left untreated
7. Suggest monitoring techniques to track recovery
8. Keep advice practical and actionable for farmers

Treatment Plan:"""
        return PromptTemplate(
            template=template, 
            input_variables=["context", "disease_name", "plant_type"]
        )

    def _create_prevention_prompt_template(self) -> PromptTemplate:
        """Create prompt template for disease prevention advice."""
        template = """You are an expert agricultural advisor specializing in plant disease prevention.

Based on the provided agricultural knowledge base, provide comprehensive prevention strategies.

Context Information:
{context}

Plant Type: {plant_type}

Instructions:
1. Provide general disease prevention strategies for this plant type
2. Recommend proper cultural practices (spacing, irrigation, fertilization)
3. Suggest resistant varieties if available
4. Include soil management and sanitation practices
5. Recommend preventive spraying schedules
6. Mention environmental conditions that promote plant health
7. Provide seasonal care guidelines
8. Include integrated pest management approaches
9. Keep advice practical and region-specific when possible

Prevention Guide:"""
        return PromptTemplate(
            template=template, 
            input_variables=["context", "plant_type"]
        )

    def _create_qa_chain(self, prompt_template):
        """Create QA chain with custom prompt."""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )

    def get_disease_solution(self, disease_prediction: str, include_sources: bool = True) -> Dict[str, any]:
        """
        Generate treatment solution for a detected disease.
        
        Args:
            disease_prediction (str): Disease prediction from the model (e.g., "Tomato___Early_blight")
            include_sources (bool): Whether to include source documents
            
        Returns:
            Dict containing disease info and treatment solution
        """
        try:
            # Parse disease prediction
            disease_parts = disease_prediction.split('___')
            if len(disease_parts) == 2:
                plant_type, disease_name = disease_parts
                plant_type = plant_type.replace('_', ' ').strip()
                disease_name = disease_name.replace('_', ' ').strip()
            else:
                plant_type = "Unknown"
                disease_name = disease_prediction.replace('_', ' ').strip()

            # Check if it's a healthy plant
            if "healthy" in disease_name.lower():
                return self.get_prevention_advice(plant_type, include_sources)

            # Generate treatment solution using direct retrieval + LLM
            query = f"treatment management {disease_name} {plant_type} plant disease control pesticide fungicide"
            
            # Get relevant documents
            docs = self.retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Format prompt with variables
            formatted_prompt = self.disease_prompt.format(
                context=context,
                disease_name=disease_name,
                plant_type=plant_type
            )
            
            # Get LLM response
            response = self.llm.invoke(formatted_prompt)
            
            result = {
                "plant_type": plant_type,
                "disease_name": disease_name,
                "prediction": disease_prediction,
                "status": "diseased",
                "solution_type": "treatment",
                "solution": response.content,
                "sources": []
            }
            
            if include_sources:
                sources = []
                for doc in docs:
                    source_info = {
                        "filename": doc.metadata.get("filename", "Unknown"),
                        "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    }
                    sources.append(source_info)
                result["sources"] = sources
            
            return result
            
        except Exception as e:
            return {
                "plant_type": "Unknown",
                "disease_name": disease_prediction,
                "prediction": disease_prediction,
                "status": "error",
                "solution_type": "error",
                "solution": f"Error generating treatment solution: {str(e)}",
                "sources": []
            }

    def get_prevention_advice(self, plant_type: str, include_sources: bool = True) -> Dict[str, any]:
        """
        Generate prevention advice for healthy plants.
        
        Args:
            plant_type (str): Type of plant
            include_sources (bool): Whether to include source documents
            
        Returns:
            Dict containing prevention advice
        """
        try:
            # Generate prevention advice using direct retrieval + LLM
            query = f"disease prevention {plant_type} plant healthy care maintenance organic farming best practices"
            
            # Get relevant documents
            docs = self.retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Format prompt with variables
            formatted_prompt = self.prevention_prompt.format(
                context=context,
                plant_type=plant_type
            )
            
            # Get LLM response
            response = self.llm.invoke(formatted_prompt)
            
            result = {
                "plant_type": plant_type,
                "disease_name": "None (Healthy)",
                "prediction": f"{plant_type}___healthy",
                "status": "healthy",
                "solution_type": "prevention",
                "solution": response.content,
                "sources": []
            }
            
            if include_sources:
                sources = []
                for doc in docs:
                    source_info = {
                        "filename": doc.metadata.get("filename", "Unknown"),
                        "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    }
                    sources.append(source_info)
                result["sources"] = sources
            
            return result
            
        except Exception as e:
            return {
                "plant_type": plant_type,
                "disease_name": "None (Healthy)",
                "prediction": f"{plant_type}___healthy",
                "status": "error",
                "solution_type": "error",
                "solution": f"Error generating prevention advice: {str(e)}",
                "sources": []
            }

    def chat_with_prediction(self, disease_prediction: str):
        """
        Interactive chat based on disease prediction.
        
        Args:
            disease_prediction (str): Disease prediction from the model
        """
        print("\n" + "="*60)
        print("🌿 Plant Disease Solution Assistant 🌿")
        print("="*60)
        
        # Get initial solution
        result = self.get_disease_solution(disease_prediction)
        
        print(f"\n🔍 Analysis Results:")
        print(f"   Plant: {result['plant_type']}")
        print(f"   Condition: {result['disease_name']}")
        print(f"   Status: {result['status'].upper()}")
        
        print(f"\n💡 {result['solution_type'].title()} Advice:")
        print(f"{result['solution']}")
        
        if result['sources']:
            print(f"\n📚 Sources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"   {i}. {source['filename']}")
        
        # Interactive follow-up
        print(f"\n🤖 Ask me follow-up questions about {result['plant_type']} care, or type 'quit' to exit.")
        
        while True:
            try:
                question = input(f"\n🧑‍🌾 You: ").strip()
                if question.lower() in ['quit', 'exit', 'bye', 'q']:
                    print(f"\n🤖 Assistant: Take care of your {result['plant_type']}! Happy farming! 🌱")
                    break
                if not question:
                    continue
                
                print(f"\n🤖 Assistant: Let me help you with that...")
                
                # Use the disease QA chain for follow-up questions
                response = self.disease_qa_chain.invoke({"query": question})
                print(f"\n{response['result']}")
                
            except KeyboardInterrupt:
                print(f"\n\n🤖 Assistant: Take care of your plants! Happy farming! 🌱")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try asking your question again.")


def main():
    """Test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Disease Detection RAG Chatbot")
    parser.add_argument("--db_path", type=str, default="./chroma_db")
    parser.add_argument("--model", type=str, default=None, help="Optional: 'gemini' or 'llama2'")
    parser.add_argument("--disease", type=str, required=True, help="Disease prediction to analyze")
    
    args = parser.parse_args()
    
    try:
        chatbot = DiseaseDetectionChatbot(
            db_path=args.db_path,
            model_name=args.model
        )
        chatbot.chat_with_prediction(args.disease)
    except Exception as e:
        print(f"❌ Error initializing chatbot: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
