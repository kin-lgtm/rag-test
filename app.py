import os
from dotenv import load_dotenv
import openai
from setup_vectorstore import VectorStore

load_dotenv()

class RAGSystem:
    def __init__(self):
        self.vector_store = VectorStore()
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def retrieve_context(self, query, n_results=3):
        """Retrieve relevant context from vector store"""
        results = self.vector_store.search(query, n_results)
        
        context_parts = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            context_parts.append(f"Source: {metadata['filename']}\n{doc}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def generate_answer(self, query, context):
        """Generate answer using LLM"""
        prompt = f"""Use the following context to answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def query(self, question):
        """Main query function"""
        print(f"\n🔍 Question: {question}\n")
        
        # Retrieve relevant context
        context = self.retrieve_context(question)
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        print(f"💡 Answer: {answer}\n")
        return answer

def main():
    rag = RAGSystem()
    
    # Interactive loop
    print("=== RAG System Ready ===")
    print("Ask questions (type 'quit' to exit)\n")
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        rag.query(question)

if __name__ == "__main__":
    main()