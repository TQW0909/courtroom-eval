from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

def get_model(model_name: str):
    local_models = {"llama3:8b-instruct-q4_K_M", "mistral:7b-instruct-q4_K_M ", "qwen2.5:14b-instruct-q4_K_M"} # TODO: Update with actual local models 
    if model_name in local_models:
        return ChatOllama(model=model_name, temperature=0.2)
    else:
        return ChatOpenAI(model=model_name, temperature=0.2)  # gpt-4o etc. 