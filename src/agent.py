from retriever import StudyRetriever
from llm_client import OllamaClient
from typing import List, Dict

class StudyCopilotAgent:
    def __init__(self):
        self.retriever = StudyRetriever()
        self.llm = OllamaClient()
        print(">>>Study Copilot Agent initialized<<<")
    
    def ask(self, query: str) -> str:
        #general query tool
        from rag_qa import rag_qa
        return rag_qa(query, self.retriever, self.llm)
    
    def generate_quiz(self, topic: str, n_questions: int = 5) -> str:
        #quiz generation tool
        chunks = self.retriever.retrieve(topic, k=8)
        context = self._build_context(chunks)
        
        system_prompt = f"""You are a quiz generator. Create {n_questions} multiple-choice questions 
        using ONLY the provided context from user's ML notes.
        
        Format: Q1: question? A) opt1 B) opt2 C) opt3 D) opt4 [Answer: X]"""
        
        prompt = f"Topic: {topic}\n\n{context}\n\nGenerate quiz:"
        return self.llm.chat(system_prompt, prompt)
    
    def revision_plan(self, topic: str, minutes: int) -> str:
        #revision planer tool
        chunks = self.retriever.retrieve(topic, k=6)
        context = self._build_context(chunks)
        
        system_prompt = f"""Create a {minutes}-minute revision plan for '{topic}'.
        Break into timed segments: read → quiz → review.
        Use context from user's notes."""
        
        prompt = f"{system_prompt}\n\n{context}\n\nDetailed plan:"
        return self.llm.chat(system_prompt, prompt)
    
    def _build_context(self, chunks):
        #constructs context from retrieved chunks
        context = "CONTEXT FROM YOUR NOTES:\n\n"
        for i, chunk in enumerate(chunks, 1):
            context += f"[{i}] {chunk['source']}:\n{chunk['text']}\n\n"
        return context
