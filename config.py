from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

LOCAL_MODELS = {
    "llama3:8b-instruct-q4_K_M",
    "mistral:7b-instruct-q4_K_M",
    "qwen2.5:14b-instruct-q4_K_M",
}

# Hard cap on generated tokens per agent call.  Keeps SLMs from producing
# ever-longer responses across rounds.  300 tokens ≈ 6-8 annotation lines,
# which is generous for the 2-4 lines we request.
AGENT_MAX_TOKENS = 300


def get_model(model_name: str, callbacks=None, max_tokens: int = AGENT_MAX_TOKENS):
    """
    Return a LangChain chat model.

    Parameters
    ----------
    model_name : str
        Model identifier. Names in LOCAL_MODELS are routed to Ollama;
        everything else goes to OpenAI.
    callbacks : list[BaseCallbackHandler] | None
        Optional LangChain callbacks (e.g. TokenTracker) attached to the model.
    max_tokens : int
        Hard ceiling on generated tokens per call (default: AGENT_MAX_TOKENS).
    """
    kwargs = {}
    if callbacks:
        kwargs["callbacks"] = callbacks

    if model_name in LOCAL_MODELS:
        return ChatOllama(model=model_name, temperature=0.2,
                          num_predict=max_tokens, **kwargs)
    else:
        return ChatOpenAI(model=model_name, temperature=0.2,
                          max_tokens=max_tokens, **kwargs)