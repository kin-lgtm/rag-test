import chromadb
from sentence_transformers import SentenceTransformer
from utils.document_loader import DocumentLoader
from utils.text_processor import TextProcessor

class VectorStore:
    def __init__(self, collection_name="rag_collection"):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_documents(self, chunks):
        """Add document chunks to vector store"""
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [f"doc_{i}" for i in range(len(chunks))]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✓ Added {len(chunks)} chunks to vector store")
    
    def search(self, query, n_results=5):
        """Search for relevant documents"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results

def setup_rag_system():
    """Main setup function"""
    print("=== Setting up RAG System ===\n")
    
    # 1. Load documents
    print("Step 1: Loading documents...")
    loader = DocumentLoader("data")
    documents = loader.load_all_documents()
    print(f"Loaded {len(documents)} documents\n")
    
    # 2. Process and chunk
    print("Step 2: Processing and chunking text...")
    processor = TextProcessor(chunk_size=500, chunk_overlap=50)
    chunks = processor.process_documents(documents)
    print(f"Created {len(chunks)} chunks\n")
    
    # 3. Create vector store
    print("Step 3: Creating vector store...")
    vector_store = VectorStore()
    vector_store.add_documents(chunks)
    
    print("\n✓ RAG system setup complete!")

if __name__ == "__main__":
    setup_rag_system()