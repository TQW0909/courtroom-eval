from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

LOCAL_MODELS = {
    "llama3:8b-instruct-q4_K_M",
    "mistral:7b-instruct-q4_K_M",
    "qwen2.5:14b-instruct-q4_K_M",
}


def get_model(model_name: str, callbacks=None):
    """
    Return a LangChain chat model.

    Parameters
    ----------
    model_name : str
        Model identifier. Names in LOCAL_MODELS are routed to Ollama;
        everything else goes to OpenAI.
    callbacks : list[BaseCallbackHandler] | None
        Optional LangChain callbacks (e.g. TokenTracker) attached to the model.
    """
    kwargs = {}
    if callbacks:
        kwargs["callbacks"] = callbacks

    if model_name in LOCAL_MODELS:
        return ChatOllama(model=model_name, temperature=0.2, **kwargs)
    else:
        return ChatOpenAI(model=model_name, temperature=0.2, **kwargs)