import requests
import json
import time
from typing import List, Dict

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:3b"  

class OllamaClient:
    def __init__(self):
        self.model = MODEL_NAME
        print(f"🤖: Connecting to Ollama ({self.model})...")
    
    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> str:
        # Prepare the payload for the Ollama API
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,  #To focus on facts
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result['message']['content'].strip()
        except Exception as e:
            return f"❌ Ollama error: {str(e)}"

def test_ollama():
    """Quick test"""
    client = OllamaClient()
    response = client.chat(
        system_prompt="You are a helpful assistant.",
        user_prompt="Explain SVM margin in one sentence."
    )
    print("Ollama test response:")
    print(response[:200] + "..." if len(response) > 200 else response)
    return client

if __name__ == "__main__":
    test_ollama()
