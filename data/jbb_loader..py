from datasets import load_dataset

def load_jbb():
    jbb_dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    return jbb_dataset
