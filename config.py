from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

def get_model(model_name: str):
    local_models = {"llama3:8b", "mistral:7b", "qwen2.5:14b"} # TODO: Update with actual local models 
    if model_name in local_models:
        return ChatOllama(model=model_name, temperature=0.2)
    else:
        return ChatOpenAI(model=model_name, temperature=0.2)  # gpt-4o etc. 