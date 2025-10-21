"""Load documents from various sources"""
import os
from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.schema import Document

def load_pdfs(pdf_directory: str) -> List[Document]:
    """Load all PDF files from a directory"""
    print(f"Loading PDFs from {pdf_directory}...")
    
    if not os.path.exists(pdf_directory):
        print(f"Directory {pdf_directory} does not exist. Skipping PDFs.")
        return []
    
    loader = DirectoryLoader(
        pdf_directory,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} PDF pages")
    return documents

def load_text_files(text_directory: str) -> List[Document]:
    """Load all text files from a directory"""
    print(f"Loading text files from {text_directory}...")
    
    if not os.path.exists(text_directory):
        print(f"Directory {text_directory} does not exist. Skipping text files.")
        return []
    
    loader = DirectoryLoader(
        text_directory,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} text files")
    return documents

def load_all_documents(pdf_dir: str, text_dir: str) -> List[Document]:
    """Load all documents from both directories"""
    all_docs = []
    all_docs.extend(load_pdfs(pdf_dir))
    all_docs.extend(load_text_files(text_dir))
    return all_docs