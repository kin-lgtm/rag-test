import os
from pathlib import Path
from pypdf import PdfReader

class DocumentLoader:
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder
    
    def load_txt(self, filepath):
        """Load text file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_pdf(self, filepath):
        """Load PDF file"""
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def load_all_documents(self):
        """Load all documents from data folder"""
        documents = []
        data_path = Path(self.data_folder)
        
        for file in data_path.rglob("*"):
            if file.is_file():
                try:
                    if file.suffix == ".txt":
                        content = self.load_txt(file)
                    elif file.suffix == ".pdf":
                        content = self.load_pdf(file)
                    else:
                        continue
                    
                    documents.append({
                        "content": content,
                        "source": str(file),
                        "filename": file.name
                    })
                    print(f"✓ Loaded: {file.name}")
                except Exception as e:
                    print(f"✗ Error loading {file.name}: {e}")
        
        return documents