import os
import glob
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import uuid

# Paths 
NOTES_DIR = "notes"
PDFS_DIR = "pdfs"
QA_DIR = "qa"
CHROMA_PATH = "data/chroma_db"
CHUNK_SIZE = 500 
CHUNK_OVERLAP = 50

def extract_text_from_file(file_path):
    #Extract text based on file type
    ext = Path(file_path).suffix.lower()
    
    if ext in ['.md', '.txt']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    elif ext == '.pdf':
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    
    return ""

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    #Split text into overlapping chunks
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if end >= len(text):
            break
    return chunks

def main():
    print("=== Building RAG index from your ML materials... ===")
    
    # Load embedding model
    print("=> Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="study_materials",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Find all files
    note_files = glob.glob(f"{NOTES_DIR}/*.md") + glob.glob(f"{NOTES_DIR}/*.txt")
    pdf_files = glob.glob(f"{PDFS_DIR}/*.pdf")
    qa_files = glob.glob(f"{QA_DIR}/*.md") + glob.glob(f"{QA_DIR}/*.txt") + glob.glob(f"{QA_DIR}/*.pdf")
    
    all_files = note_files + pdf_files + qa_files
    print(f"-> Found {len(all_files)} files to process:")
    for f in all_files[:5]:  # Show first 5
        print(f"   - {Path(f).name}")
    if len(all_files) > 5:
        print(f"   ... and {len(all_files)-5} more")
    
    # Process files
    total_chunks = 0
    for file_path in all_files:
        print(f"\n-> Processing {Path(file_path).name}...")
        
        text = extract_text_from_file(file_path)
        if not text.strip():
            print("   !!!  No text extracted, skipping  !!!")
            continue
            
        chunks = chunk_text(text)
        print(f"   → {len(chunks)} chunks created")
        
        # Embed chunks
        embeddings = embedder.encode(chunks).tolist()
        
        # Prepare metadata
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": Path(file_path).name, "type": "chunk"} for _ in chunks]
        
        # Add to ChromaDB
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        total_chunks += len(chunks)
    
    print(f"\n✅ SUCCESS! Vector index built with {total_chunks} chunks")
    print(f"Stored in: {CHROMA_PATH}")
    print("Ready for retrieval! Run src/retriever.py next.")

if __name__ == "__main__":
    main()
