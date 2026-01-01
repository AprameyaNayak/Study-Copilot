import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path


# Same config as ingest.py
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "study_materials"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 500


class StudyRetriever:
    def __init__(self):
        print("🔍 Loading retriever...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        print(f"Loaded {self.collection.count()} chunks from your materials")
        
        # Debug: Show what files are actually indexed
        sample_meta = self.collection.get(limit=3)['metadatas']
        print("Sample sources:", [m['source'] for m in sample_meta])
    
    def retrieve(self, query, k=5):
        # retrieves the relavent chunks for a given query
        # Embed query
        query_embedding = self.embedder.encode([query]).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        retrieved = []
        for i in range(len(results['documents'][0])):
            retrieved.append({
                'text': results['documents'][0][i],
                'source': results['metadatas'][0][i]['source'],
                'score': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        return retrieved
    
    def get_stats(self):
        #Returns resource stats
        count = self.collection.count()
        return f"Knowledge base: {count:,} chunks across {len(set([m['source'] for m in self.collection.get()['metadatas']]))} files"


def main():
    retriever = StudyRetriever()
    
    print(retriever.get_stats())
    print("\n" + "="*60)
    print("Test queries on YOUR ML materials:")
    print("="*60)
    
    test_queries = [
        "SVM margin",
        "neural network backpropagation", 
        "feature selection",
        "Security+ firewall",
        "LightGBM"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: '{query}'")
        print("-" * 40)
        results = retriever.retrieve(query, k=3)
        
        for i, chunk in enumerate(results, 1):
            print(f"   {i}. [{chunk['source']}] {chunk['text'][:150]}...")
            print(f"      Score: {chunk['score']:.3f}")
        
        print()

if __name__ == "__main__":
    main()
