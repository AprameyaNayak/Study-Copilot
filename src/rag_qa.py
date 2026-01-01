from retriever import StudyRetriever
from llm_client import OllamaClient

def build_context(retrieved_chunks):
    context = "CONTEXT FROM YOUR NOTES:\n\n"
    for i, chunk in enumerate(retrieved_chunks, 1):
        context += f"[{i}] {chunk['source']}:\n{chunk['text']}\n\n"
    return context

def rag_qa(query, retriever, llm):
    chunks = retriever.retrieve(query, k=10)
    context = build_context(chunks)
    
    system_prompt = """You are a study copilot using ONLY the provided context from user's ML notes.
Answer accurately. Cite sources [1], [2] etc. from context.
If not in context, say "Not found in your materials."""
    
    user_prompt = f"Question: {query}\n\n{context}\nAnswer:"
    return llm.chat(system_prompt, user_prompt)

def main():
    print("🎓 RAG Q&A: Your 1999 ML chunks + Ollama")
    print("=" * 50)
    
    retriever = StudyRetriever()
    llm = OllamaClient()
    
    queries = ["SVM margin", "backpropagation", "feature selection"]
    
    for query in queries:
        print(f"\n❓ Q: {query}")
        print("-" * 40)
        answer = rag_qa(query, retriever, llm)
        print(answer)
        print()

if __name__ == "__main__":
    main()
