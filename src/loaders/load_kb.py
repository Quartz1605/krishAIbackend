import os
from typing import List, Dict
from pathlib import Path
import pypdf
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFKnowledgeLoader:
    """
    Loads and processes PDF files from the knowledge base directory.
    """
    
    def __init__(self, kb_directory: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF loader.
        
        Args:
            kb_directory (str): Path to the directory containing PDF files
            chunk_size (int): Size of text chunks for processing
            chunk_overlap (int): Overlap between consecutive chunks
        """
        self.kb_directory = Path(kb_directory)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path (Path): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error reading {pdf_path}: {str(e)}")
        
        return text
    
    def load_pdf_documents(self) -> List[Document]:
        """
        Load all PDF documents from the knowledge base directory.
        
        Returns:
            List[Document]: List of LangChain Document objects
        """
        documents = []
        
        if not self.kb_directory.exists():
            print(f"Knowledge base directory {self.kb_directory} does not exist!")
            return documents
        
        # Find all PDF files in the directory
        pdf_files = list(self.kb_directory.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.kb_directory}")
            return documents
        
        print(f"Found {len(pdf_files)} PDF files to process...")
        
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file.name}...")
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_file)
            
            if text.strip():
                # Create document with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_file),
                        "filename": pdf_file.name,
                        "type": "pdf"
                    }
                )
                documents.append(doc)
                print(f"Successfully loaded {pdf_file.name} ({len(text)} characters)")
            else:
                print(f"Warning: No text extracted from {pdf_file.name}")
        
        return documents
    
    def load_and_split_documents(self) -> List[Document]:
        """
        Load PDF documents and split them into chunks.
        
        Returns:
            List[Document]: List of chunked Document objects
        """
        # Load all documents
        documents = self.load_pdf_documents()
        
        if not documents:
            print("No documents loaded!")
            return []
        
        print(f"Loaded {len(documents)} documents. Now splitting into chunks...")
        
        # Split documents into chunks
        chunked_docs = self.text_splitter.split_documents(documents)
        
        print(f"Created {len(chunked_docs)} text chunks")
        
        return chunked_docs
    
    def load_text_file(self, file_path: Path) -> Document:
        """
        Load a text file as a Document.
        
        Args:
            file_path (Path): Path to the text file
            
        Returns:
            Document: LangChain Document object
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            return Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "type": "text"
                }
            )
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            return None
    
    def load_all_documents(self) -> List[Document]:
        """
        Load all documents (PDF and text) from the knowledge base directory.
        
        Returns:
            List[Document]: List of chunked Document objects
        """
        documents = []
        
        # Load PDF documents
        pdf_docs = self.load_pdf_documents()
        documents.extend(pdf_docs)
        
        # Load text files
        text_files = list(self.kb_directory.glob("*.txt"))
        for text_file in text_files:
            print(f"Processing {text_file.name}...")
            doc = self.load_text_file(text_file)
            if doc:
                documents.append(doc)
                print(f"Successfully loaded {text_file.name}")
        
        if not documents:
            print("No documents loaded!")
            return []
        
        print(f"Loaded {len(documents)} documents. Now splitting into chunks...")
        
        # Split all documents into chunks
        chunked_docs = self.text_splitter.split_documents(documents)
        
        print(f"Created {len(chunked_docs)} text chunks")
        
        return chunked_docs


if __name__ == "__main__":
    # Test the loader
    kb_path = "../../data/kb"
    loader = PDFKnowledgeLoader(kb_path)
    documents = loader.load_all_documents()
    
    print(f"\nLoaded {len(documents)} document chunks")
    if documents:
        print(f"Sample chunk: {documents[0].page_content[:200]}...")
