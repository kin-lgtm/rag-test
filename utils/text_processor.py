class TextProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove extra whitespace
        text = " ".join(text.split())
        return text.strip()
    
    def chunk_text(self, text, metadata=None):
        """Split text into overlapping chunks"""
        text = self.clean_text(text)
        chunks = []
        
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > 0:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunk_data = {
                "text": chunk.strip(),
                "metadata": metadata or {}
            }
            chunks.append(chunk_data)
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_documents(self, documents):
        """Process multiple documents into chunks"""
        all_chunks = []
        
        for doc in documents:
            metadata = {
                "source": doc["source"],
                "filename": doc["filename"]
            }
            chunks = self.chunk_text(doc["content"], metadata)
            all_chunks.extend(chunks)
        
        return all_chunks