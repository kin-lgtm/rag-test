"""
Main FastAPI server for RAG chatbot
Run with: python app.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config.settings import settings
import uvicorn

# Initialize FastAPI
app = FastAPI(title="Agriculture RAG API")

# CORS - Allow your React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    success: bool
    sources: list = []

# Initialize RAG system
print("🚀 Loading RAG system...")

embeddings = OpenAIEmbeddings(
    model=settings.EMBEDDING_MODEL,
    openai_api_key=settings.OPENAI_API_KEY
)

vectorstore = Chroma(
    persist_directory=settings.VECTORSTORE_PATH,
    embedding_function=embeddings
)

# Custom prompt template
PROMPT_TEMPLATE = """You are a helpful agricultural consultant assistant for Asian Agricultural Consultancies, a leading agricultural consultancy in Sri Lanka.

Use the following context from our documentation to answer the question. 

Context:
{context}

Question: {question}

Guidelines:
- Provide accurate, helpful answers based on the context
- If you're not sure, say so and suggest contacting us
- Be professional but friendly
- Mention specific services when relevant
- Encourage users to contact us for personalized consultation

Answer:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

llm = ChatOpenAI(
    temperature=settings.TEMPERATURE,
    model=settings.CHAT_MODEL,
    openai_api_key=settings.OPENAI_API_KEY
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 4}  # Retrieve top 4 relevant chunks
    ),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

print("✅ RAG system loaded successfully!")

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Agriculture RAG API is running",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        # Get answer from RAG system
        result = qa_chain({"query": request.question})
        
        # Extract sources
        sources = []
        if "source_documents" in result:
            sources = [
                {
                    "content": doc.page_content[:200],  # First 200 chars
                    "source": doc.metadata.get("source", "unknown")
                }
                for doc in result["source_documents"][:2]  # Top 2 sources
            ]
        
        return ChatResponse(
            answer=result["result"],
            success=True,
            sources=sources
        )
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Sorry, I encountered an error. Please try again."
        )

# Run server
if __name__ == "__main__":
    print(f"\n🌐 Starting server at http://{settings.API_HOST}:{settings.API_PORT}")
    print("📡 API docs available at http://localhost:8000/docs")
    
    uvicorn.run(
        "app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True  # Auto-reload on code changes
    )