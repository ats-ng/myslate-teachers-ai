from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
import requests
from io import BytesIO
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import logging
from urllib.parse import urlparse
import re
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Generic AI Content Generator")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
AI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize AI client with proper error handling
ai_client = None
if AI_API_KEY:
    ai_client = OpenAI(api_key=AI_API_KEY)
else:
    logger.warning("OPENAI_API_KEY not found. AI features will be disabled.")

# Load embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Request models
class DocumentRequest(BaseModel):
    urls: List[str]

class GenerationRequest(BaseModel):
    prompt: str
    document_urls: Optional[List[str]] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2000

# Helper functions
async def download_and_extract_text(url: str) -> str:
    """Download file from URL and extract text content"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        content = response.content
        file_extension = urlparse(url).path.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
            text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        elif file_extension in ['docx', 'doc']:
            doc = docx.Document(BytesIO(content))
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        elif file_extension == 'txt':
            text = content.decode('utf-8')
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        return text[:10000]  # Limit text length to avoid context overload
        
    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Could not process document: {str(e)}")

async def process_documents(document_urls: List[str]) -> List[str]:
    """Process multiple documents and return their text content"""
    document_texts = []
    for url in document_urls:
        try:
            text = await download_and_extract_text(url)
            if text.strip():
                document_texts.append(text)
        except Exception as e:
            logger.warning(f"Skipping document {url}: {e}")
            continue
    return document_texts

def create_vector_index(documents: List[str]):
    """Create FAISS index from document texts"""
    if not documents:
        return None
    
    embeddings = embedding_model.encode(documents)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_relevant_content(index, documents: List[str], query: str, k: int = 3) -> str:
    """Search for relevant content in documents and return as context string"""
    if index is None or not documents:
        return ""
    
    query_embedding = embedding_model.encode([query])
    D, I = index.search(query_embedding, k)
    
    relevant_content = []
    for idx in I[0]:
        if 0 <= idx < len(documents):
            content = documents[idx]
            sentences = re.split(r'[.!?]+', content)
            if sentences:
                relevant_sentences = sentences[:2]
                relevant_content.append(' '.join([s.strip() for s in relevant_sentences if s.strip()]))
    
    if relevant_content:
        return "Relevant information from provided documents:\n" + "\n".join(relevant_content)
    return ""

async def call_ai_provider(prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """Call OpenAI API for content generation"""
    if not ai_client:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
    
    try:
        response = ai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using gpt-3.5-turbo as fallback, you can change to gpt-4 if available
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"AI API error: {e}")
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

# API Endpoints
@app.post("/generate-content/")
async def generate_content(request: GenerationRequest = Body(...)):
    """
    Generate content using AI with optional RAG from documents
    """
    try:
        rag_context = ""
        
        # Process documents if provided
        if request.document_urls:
            document_texts = await process_documents(request.document_urls)
            if document_texts:
                index = create_vector_index(document_texts)
                rag_context = search_relevant_content(index, document_texts, request.prompt)
        
        # Build final prompt with RAG context
        final_prompt = request.prompt
        
        if rag_context:
            final_prompt = f"""
            {request.prompt}
            
            CONTEXT FROM PROVIDED DOCUMENTS:
            {rag_context}
            
            Please incorporate relevant information from the provided documents into your response.
            """
        
        # Generate content
        generated_content = await call_ai_provider(
            final_prompt, 
            temperature=request.temperature, 
            max_tokens=request.max_tokens
        )
        
        return JSONResponse(content={
            "success": True,
            "generated_content": generated_content,
            "metadata": {
                "documents_processed": len(request.document_urls) if request.document_urls else 0,
                "rag_context_used": bool(rag_context),
                "prompt_length": len(request.prompt),
                "response_length": len(generated_content)
            }
        })
        
    except Exception as e:
        logger.error(f"Content generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-documents/")
async def process_documents_endpoint(request: DocumentRequest = Body(...)):
    """
    Process documents and return extracted text (for testing/document preview)
    """
    try:
        document_texts = await process_documents(request.urls)
        
        return JSONResponse(content={
            "success": True,
            "documents_processed": len(document_texts),
            "total_documents": len(request.urls),
            "extracted_texts": [
                {
                    "url": request.urls[i],
                    "text_preview": text[:500] + "..." if len(text) > 500 else text,
                    "length": len(text)
                }
                for i, text in enumerate(document_texts)
            ]
        })
        
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    ai_status = "enabled" if ai_client else "disabled"
    return {
        "status": "healthy", 
        "service": "AI Content Generator",
        "ai_features": ai_status,
        "embedding_model": EMBEDDING_MODEL
    }

@app.get("/")
async def root():
    ai_status = "enabled" if ai_client else "disabled - set OPENAI_API_KEY environment variable"
    return {
        "message": "Generic AI Content Generator API", 
        "version": "1.0",
        "ai_status": ai_status,
        "endpoints": {
            "POST /generate-content": "Generate content with optional RAG",
            "POST /process-documents": "Preview document content extraction",
            "GET /health": "Service health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)