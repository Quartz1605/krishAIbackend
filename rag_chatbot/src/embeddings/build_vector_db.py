import os
import sys
from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from loaders.load_kb import PDFKnowledgeLoader

class VectorDBBuilder:
    """
    Builds and manages ChromaDB vector database for the RAG system.
    """
    
    def __init__(self, 
                 db_path: str = "./chroma_db", 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 collection_name: str = "kerala_farming_kb"):
        """
        Initialize the Vector DB Builder.
        
        Args:
            db_path (str): Path to store the ChromaDB database
            embedding_model (str): Hugging Face embedding model name
            collection_name (str): Name for the ChromaDB collection
        """
        self.db_path = Path(db_path)
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        
        # Initialize embeddings
        print(f"Loading embedding model: {embedding_model}", file=sys.stderr)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create db directory if it doesn't exist
        self.db_path.mkdir(parents=True, exist_ok=True)
        
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        Create a ChromaDB vector store from documents.
        
        Args:
            documents (List[Document]): List of document chunks
            
        Returns:
            Chroma: ChromaDB vector store
        """
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        print(f"Creating vector store with {len(documents)} documents...")
        
        # Create ChromaDB vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.db_path),
            collection_name=self.collection_name
        )
        
        # Persist the database
        vectorstore.persist()
        print(f"Vector store created and persisted at {self.db_path}")
        
        return vectorstore
    
    def load_existing_vector_store(self) -> Optional[Chroma]:
        """
        Load an existing ChromaDB vector store.
        
        Returns:
            Optional[Chroma]: ChromaDB vector store if exists, None otherwise
        """
        try:
            if self.db_path.exists():
                print(f"Loading existing vector store from {self.db_path}", file=sys.stderr)
                vectorstore = Chroma(
                    persist_directory=str(self.db_path),
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
                return vectorstore
            else:
                print(f"No existing vector store found at {self.db_path}", file=sys.stderr)
                return None
        except Exception as e:
            print(f"Error loading vector store: {str(e)}", file=sys.stderr)
            return None
    
    def build_or_load_vector_db(self, kb_directory: str, force_rebuild: bool = False) -> Chroma:
        """
        Build or load the vector database.
        
        Args:
            kb_directory (str): Path to knowledge base directory
            force_rebuild (bool): Whether to force rebuild even if DB exists
            
        Returns:
            Chroma: ChromaDB vector store
        """
        # Check if vector store already exists
        if not force_rebuild:
            existing_store = self.load_existing_vector_store()
            if existing_store is not None:
                print("Using existing vector database")
                return existing_store
        
        print("Building new vector database...")
        
        # Load documents from knowledge base
        loader = PDFKnowledgeLoader(kb_directory)
        documents = loader.load_all_documents()
        
        if not documents:
            raise ValueError(f"No documents found in {kb_directory}")
        
        # Create vector store
        vectorstore = self.create_vector_store(documents)
        
        return vectorstore
    
    def add_documents_to_existing_store(self, documents: List[Document]) -> None:
        """
        Add new documents to an existing vector store.
        
        Args:
            documents (List[Document]): List of new documents to add
        """
        vectorstore = self.load_existing_vector_store()
        
        if vectorstore is None:
            raise ValueError("No existing vector store found. Create one first.")
        
        print(f"Adding {len(documents)} documents to existing vector store...")
        vectorstore.add_documents(documents)
        vectorstore.persist()
        print("Documents added successfully")
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents in the vector store.
        
        Args:
            query (str): Search query
            k (int): Number of similar documents to retrieve
            
        Returns:
            List[Document]: List of similar documents
        """
        vectorstore = self.load_existing_vector_store()
        
        if vectorstore is None:
            raise ValueError("No vector store found. Build one first.")
        
        # Perform similarity search
        similar_docs = vectorstore.similarity_search(query, k=k)
        
        return similar_docs
    
    def get_retriever(self, k: int = 5):
        """
        Get a retriever for the vector store.
        
        Args:
            k (int): Number of documents to retrieve
            
        Returns:
            VectorStoreRetriever: Retriever object
        """
        vectorstore = self.load_existing_vector_store()
        
        if vectorstore is None:
            raise ValueError("No vector store found. Build one first.")
        
        return vectorstore.as_retriever(search_kwargs={"k": k})


def main():
    """
    Main function to build the vector database.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Build ChromaDB vector database from PDF knowledge base")
    parser.add_argument("--kb_dir", type=str, default="../../data/kb", 
                       help="Path to knowledge base directory")
    parser.add_argument("--db_path", type=str, default="./chroma_db",
                       help="Path to store ChromaDB database")
    parser.add_argument("--force_rebuild", action="store_true",
                       help="Force rebuild even if database exists")
    
    args = parser.parse_args()
    
    try:
        # Initialize vector DB builder
        builder = VectorDBBuilder(db_path=args.db_path)
        
        # Build or load vector database
        vectorstore = builder.build_or_load_vector_db(
            kb_directory=args.kb_dir,
            force_rebuild=args.force_rebuild
        )
        
        print("\n=== Vector Database Ready ===")
        print(f"Database path: {args.db_path}")
        print(f"Collection: {builder.collection_name}")
        
        # Test search functionality
        test_query = "pest control in rice cultivation"
        print(f"\nTesting search with query: '{test_query}'")
        similar_docs = builder.search_similar_documents(test_query, k=3)
        
        print(f"Found {len(similar_docs)} relevant documents:")
        for i, doc in enumerate(similar_docs):
            print(f"\n{i+1}. Source: {doc.metadata.get('filename', 'Unknown')}")
            print(f"   Preview: {doc.page_content[:150]}...")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
