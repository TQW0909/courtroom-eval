from datasets import load_dataset

def load_jbb():
    jbb_dataset_behaviors = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    
    jbb_dataset_judge_comparison = load_dataset("JailbreakBench/JBB-Behaviors", "judge_comparison")

    return jbb_dataset_behaviors, jbb_dataset_judge_comparison
