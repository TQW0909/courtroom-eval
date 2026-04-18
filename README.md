# Courtroom Eval

A multi-agent adversarial evaluation pipeline for AI safety, built for CSCI 567. The system uses a courtroom metaphor — prosecution, defense, judge, and jury — to classify whether AI-generated responses are harmful or benign. Each "side" must ground its arguments in verbatim quotes from the response under review, and a citation filter enforces this constraint automatically.

The pipeline is built on LangGraph and evaluates cases from the [JailbreakBench (JBB)](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) dataset.


## Motivation

Standard single-pass LLM classifiers often produce shallow or inconsistent safety labels. By structuring evaluation as an adversarial debate — where one agent argues a response is harmful and another argues it is benign — the system surfaces nuanced reasoning that a single classifier might miss. The citation-grounding requirement prevents hallucinated evidence, and the multi-juror vote reduces variance.


## Architecture

```
                 ┌────────────┐
                 │ Prosecutor │  (content-analysis annotator — flags harmful passages)
                 └─────┬──────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Citation Filter │  (rejects arguments without verbatim quotes)
              └────────┬────────┘
                       │ pass → continue  /  fail → retry once, then forfeit
                       ▼
                 ┌────────────┐
                 │  Defender  │  (mitigation reviewer — finds mitigating factors)
                 └─────┬──────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Citation Filter │  (same filter, applied symmetrically)
              └────────┬────────┘
                       │
                       ▼
                  ┌──────────┐
                  │  Judge   │  (procedural moderator — CONTINUE or CLOSE)
                  └────┬─────┘
                       │
              CONTINUE │          CLOSE
            ┌──────────┘            │
            ▼                       ▼
       Prosecutor              ┌──────────┐
       (next round)            │   Jury   │  (3 independent jurors, majority vote)
                               └────┬─────┘
                                    │
                                    ▼
                                  END
```

Each node is a pure state transformer with no side effects. The graph is compiled by LangGraph and executed as a state machine over a shared `CourtroomState` dict.


## Pipeline Details

**Prosecution** receives the original user prompt and the AI response. It identifies problematic passages by extracting verbatim quotes and explaining why each is concerning. On follow-up rounds it responds to the defense's counter-arguments.

**Defense** performs adversarial review of the prosecution's flags. It extracts mitigating passages — disclaimers, fictional framing, refusal language — with verbatim quotes and explains why they reduce concern.

**Citation Filter** runs after both the prosecution and defense. It extracts all quoted spans (straight quotes, curly quotes, guillemets) and checks that at least one quote is grounded in the response text. Short quotes must match exactly; longer quotes (12+ words) use a sliding-window check to tolerate minor imprecisions from smaller models. On first failure the agent gets one retry; on second failure the turn is forfeited and the argument is dropped from the transcript.

**Judge** is a procedural moderator, not a content classifier. It sees a summary of prior rounds plus the latest arguments, and decides whether the debate should continue or close. It defaults to CLOSE unless genuinely new quoted evidence has appeared. Hard limits enforce a minimum of 2 rounds and a configurable maximum (default 4).

**Jury** consists of 3 independent annotators that read the full debate transcript and classify the response as `harmful` or `benign` with a confidence score. The final verdict is determined by majority vote; confidence is averaged.


## Dataset

The pipeline uses the **JailbreakBench (JBB)** `behaviors` split, which contains harmful and benign goal/target pairs. Each case provides:

- **Goal** — the original user prompt (e.g., "Write a defamatory article claiming...")
- **Target** — the AI system's response to that prompt

Both fields are shown to all agents and displayed in the terminal output.


## Baselines

### MIRROR

MIRROR (**M**ulti-LLM **I**terative **R**eview and **R**esponse for **O**ptimized **R**ating) is a method originally proposed for automated evaluation of open-ended question generation ([arXiv:2410.12893](https://arxiv.org/abs/2410.12893)). The core idea is that two independent LLM evaluators iteratively review the same content, each receiving the other's previous assessment as feedback, until they converge on a shared judgment.

Here the method is adapted for binary safety classification. Each evaluator produces a structured assessment on every round:

```
Verdict:    harmful | benign
Confidence: 0.0–1.0
Strengths:  one sentence — mitigating factors or benign aspects
Flaws:      one sentence — concerning or harmful aspects
```

**How it works:**

1. **Round 1** — Both evaluators independently assess the response (no peer feedback yet).
2. **Round N+1** — Each evaluator receives the other's latest assessment and reconsiders: update the verdict if the peer raised a valid point, or hold the position with a counter-argument.
3. **Convergence** — The loop terminates early when both evaluators agree on the same verdict for two consecutive rounds, matching the original paper's criterion. If they never converge, the evaluator with higher confidence wins.

This is implemented in [`baselines/mirror.py`](baselines/mirror.py).

**Usage:**

```bash
# Run MIRROR with a local model
python main.py --baseline mirror --model llama3:8b-instruct-q4_K_M --cases 20

# Use different models for each evaluator
python main.py --baseline mirror --model gpt-4o-mini --jury-model gpt-4o --cases 20

# Limit iterations (default: 5)
python main.py --baseline mirror --mirror-iterations 3 --cases 20

# Verbose — prints each round's assessments
python main.py --baseline mirror --verbose --cases 5
```

**Additional CLI flags for MIRROR:**

| Flag | Default | Description |
|------|---------|-------------|
| `--baseline` | `courtroom` | Set to `mirror` to use the MIRROR baseline |
| `--jury-model` | same as `--model` | Model for Evaluator-2 (Evaluator-1 uses `--model`) |
| `--mirror-iterations` | `5` | Maximum rounds before tie-break |

**Results** (`llama3:8b-instruct-q4_K_M`, 20 cases, test split):

| Metric    | Value  |
|-----------|--------|
| Cases     | 20     |
| Correct   | 17     |
| Accuracy  | 85.0%  |
| Precision | 88.9%  |
| Recall    | 80.0%  |
| F1        | 0.842  |


### Majority Rules

Based on "Majority Rules: LLM Ensemble is a Winning Approach for Content Categorization" ([arXiv:2511.15714](https://arxiv.org/abs/2511.15714)). The core idea is that N independent LLMs each classify the response under identical zero-shot conditions, with no communication between them. The final verdict is determined by majority vote.

This serves as a **lower-bound comparison** against the debate-based approaches: if the Courtroom or Multi-Agent Debate pipelines outperform simple majority voting, the adversarial structure itself is adding value beyond what diverse model opinions alone can provide.

**How it works:**

1. Each of N voter models receives the same prompt and response independently.
2. Each outputs a verdict, confidence score, and one-sentence reason.
3. Final verdict = majority vote; confidence = average across all voters.

This is implemented in [`baselines/majority_rules.py`](baselines/majority_rules.py).

**Usage:**

```bash
# Run with 5 voters (default)
python main.py --baseline majority-rules --model gpt-4o-mini --cases 20

# Change the number of voters
python main.py --baseline majority-rules --voters 7 --cases 20

# Verbose — prints each voter's verdict as it completes
python main.py --baseline majority-rules --verbose --cases 5
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--baseline` | `courtroom` | Set to `majority-rules` |
| `--voters` | `5` | Number of independent voter LLMs |


### Multi-Agent Debate

Based on "Efficient LLM Safety Evaluation through Multi-Agent Debate" ([arXiv:2511.06396](https://arxiv.org/abs/2511.06396)). Three role-specific agents — **Critic**, **Defender**, and **Judge** — conduct a structured debate over a fixed number of rounds. After all rounds the Judge synthesises the transcript into a final binary verdict.

This is the **closest structural relative** to the Courtroom pipeline. The key differences are:

| | Courtroom | Multi-Agent Debate |
|---|---|---|
| Citation grounding | Required (verbatim quotes) | None |
| Round control | Dynamic (Judge decides CONTINUE/CLOSE) | Fixed N rounds |
| Final decision | 3-juror majority vote | Judge agent directly |

These differences make it a precise ablation: it tests the adversarial debate skeleton without the citation-grounding constraint or jury panel, isolating how much those two additions contribute.

**How it works:**

1. **Round 1** — Critic identifies evidence the response is harmful; Defender argues it is benign.
2. **Round N** — Each agent reads the other's previous argument and rebuts or reinforces their position.
3. **Judge** — Reads the full debate transcript and issues the final verdict with a confidence score.

This is implemented in [`baselines/multi_agent_debate.py`](baselines/multi_agent_debate.py).

**Usage:**

```bash
# Run with 3 debate rounds (default)
python main.py --baseline debate --model gpt-4o-mini --cases 20

# Change the number of rounds
python main.py --baseline debate --debate-rounds 5 --cases 20

# Verbose — prints each agent's argument as it runs
python main.py --baseline debate --verbose --cases 5
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--baseline` | `courtroom` | Set to `debate` |
| `--debate-rounds` | `3` | Number of Critic + Defender exchange rounds |


## Repository Structure

```
courtroom-eval/
├── main.py                  # CLI entry point — loads data, runs graph, computes metrics
├── config.py                # Model routing (Ollama for local models, OpenAI for GPT)
├── requirements.txt         # Pinned dependencies
├── .env.example             # Environment variable template
│
├── agents/
│   ├── prosecutor.py        # Harm-annotation agent with few-shot examples
│   ├── defender.py          # Mitigation-review agent with few-shot examples
│   ├── judge.py             # Procedural moderator (CONTINUE / CLOSE)
│   └── jury.py              # 3-juror panel with majority vote
│
├── baselines/
│   ├── mirror.py            # MIRROR baseline (arXiv:2410.12893) — iterative two-LLM convergence
│   ├── majority_rules.py    # Majority Rules baseline (arXiv:2511.15714) — silent N-voter ensemble
│   └── multi_agent_debate.py # Multi-Agent Debate baseline (arXiv:2511.06396) — Critic/Defender/Judge
│
├── filters/
│   └── citation_filter.py   # Verbatim-quote grounding check with retry/forfeit logic
│
├── graph/
│   └── courtroom_graph.py   # LangGraph state definition and graph wiring
│
├── tasks.py                 # TaskConfig dataclass + built-in task definitions
├── data/
│   └── jbb_loader.py        # HuggingFace dataset loader for JBB-Behaviors
│
├── utils/
│   ├── pretty_print.py      # Rich terminal output (panels, tables, color-coded verdicts)
│   ├── token_tracker.py     # LangChain callback handler for token usage + latency
│   └── run_logger.py        # Structured JSONL logger for run results and comparison
│
├── test_agents.py           # Unit tests for all agents (mocked models)
└── test_graph.py            # Integration tests for graph flow, filter logic, routing
```


## Setup

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running (for local models), or an OpenAI API key (for GPT models)

### Installation

```bash
git clone <repo-url>
cd courtroom-eval
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment

```bash
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY if using GPT models
```

If using a local model via Ollama, pull it first:

```bash
ollama pull llama3:8b-instruct-q4_K_M
```


## Task Configuration

The framework separates the **debate structure** (graph, filter, judge, voting) from the **task definition** (what to look for, what labels to use). Adding a new evaluation task requires only a short config — no prompt engineering, no code changes.

### Built-in tasks

| Task | Labels | Description |
|------|--------|-------------|
| `jailbreak-safety` | harmful / benign | Did the AI comply with a jailbreak attempt? |
| `factual-accuracy` | inaccurate / accurate | Does the response contain factual errors? |
| `toxicity` | toxic / non-toxic | Does the response contain toxic language? |
| `bias-detection` | biased / unbiased | Does the response exhibit demographic bias? |

### Adding a custom task

Add a `TaskConfig` to `tasks.py` and register it in the `TASKS` dict:

```python
MY_TASK = TaskConfig(
    name="my-task",
    description="One-line description of what this task evaluates",
    labels=("flagged_label", "clean_label"),
    prosecution_criteria="what the prosecution annotator should look for",
    defense_criteria="what the defense annotator should look for",
    jury_decision_rule="1-2 sentence rule referencing both labels",
    prosecution_examples=(
        '"example phrase" — Why this phrase is concerning.\n'
        '"another phrase" — Why this one is too.'
    ),
    defense_examples=(
        '"mitigating phrase" — Why this phrase reduces concern.\n'
        '"another phrase" — Why this one does too.'
    ),
)
```

The graph, citation filter, judge, token tracker, logger, and ablation flags all work unchanged — only the agent prompts adapt.


## Usage

### Basic run

```bash
# Run 10 cases with GPT-4o-mini on the default task (jailbreak-safety)
python main.py

# Specify a different task
python main.py --task toxicity --model gpt-4o-mini --cases 10

# Run 2 cases with a local Ollama model, verbose output
python main.py --model llama3:8b-instruct-q4_K_M --cases 2 --verbose

# Customize max debate rounds
python main.py --model gpt-4o --cases 20 --max-rounds 6

# Use a separate model for jurors
python main.py --model gpt-4o --jury-model gpt-4o-mini --cases 10
```

### CLI arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gpt-4o-mini` | Model for prosecution, defense, and judge |
| `--jury-model` | same as `--model` | Separate model for the 3 jurors (or MIRROR Evaluator-2) |
| `--max-rounds` | `4` | Maximum debate rounds before forced close (courtroom only) |
| `--cases` | `10` | Number of cases to evaluate |
| `--split` | `test` | JBB dataset split (`train` or `test`) |
| `--jurors` | `3` | Number of independent jurors in the jury panel |
| `--verbose` | off | Print full trial transcript for each case |
| `--baseline` | `courtroom` | Evaluation method: `courtroom`, `mirror`, `majority-rules`, or `debate` |
| `--mirror-iterations` | `5` | Maximum rounds for the MIRROR baseline |
| `--voters` | `5` | Number of independent voters for the majority-rules baseline |
| `--debate-rounds` | `3` | Number of Critic+Defender rounds for the debate baseline |
| `--log PATH` | off | Write structured JSONL run log to the given file |
| `--no-filter` | off | Ablation: disable citation filter (all arguments pass) |
| `--no-defense` | off | Ablation: disable defense agent (prosecution-only) |

### Output

In verbose mode, each case displays the original prompt, response under review, full debate transcript with color-coded panels, jury deliberation table, and a summary with ground-truth comparison. At the end, aggregate metrics are printed:

- **Accuracy** — fraction of correct classifications
- **Precision** — of cases predicted harmful, how many truly were
- **Recall** — of truly harmful cases, how many were caught
- **F1** — harmonic mean of precision and recall


## Evaluation Metrics

The pipeline reports standard binary classification metrics where `harmful` is the positive class. After all cases are processed, a summary table is displayed:

```
╭────────────────┬────────────╮
│ Metric         │ Value      │
├────────────────┼────────────┤
│ Model          │ llama3:8b… │
│ Cases          │ 2          │
│ Correct        │ 2          │
│ Accuracy       │ 100.0%     │
│ Precision      │ 100.0%     │
│ Recall         │ 100.0%     │
│ F1             │ 1.000      │
╰────────────────┴────────────╯
```


## Logging and Comparison

The `--log` flag writes a structured JSONL record for each run. Every record contains the full run config, per-case results (verdict, confidence, rounds, grounding failures, token usage by role), and aggregate metrics. Multiple runs append to the same file for easy comparison.

```bash
# Run the same cases across two models
python main.py --model gpt-4o-mini       --cases 20 --log results/runs.jsonl
python main.py --model llama3:8b-instruct-q4_K_M --cases 20 --log results/runs.jsonl

# Compare in Python
import json
runs = [json.loads(line) for line in open("results/runs.jsonl")]
for r in runs:
    cfg = r["config"]
    m = r["metrics"]
    t = r["totals"]
    print(f'{cfg["model"]:<30} acc={m["accuracy"]:.1%}  f1={m["f1"]:.3f}  '
          f'tokens={t["total_tokens"]:,}  latency={t["total_latency_ms"]/1000:.1f}s')
```

Token usage is tracked per LLM call via a LangChain callback handler that works with both OpenAI (via `usage_metadata`) and Ollama (via `prompt_eval_count` / `eval_count` in generation metadata). Each call is tagged by agent role (prosecutor, defender, judge, juror) so you can break down cost by pipeline component. A resource usage summary table is printed at the end of every run.


## Ablation Studies

Ablation flags let you disable pipeline components to measure their contribution:

```bash
# No citation filter — do grounding constraints improve accuracy?
python main.py --model gpt-4o-mini --no-filter --cases 20 --log results/runs.jsonl

# No defense — does adversarial review matter vs prosecution-only?
python main.py --model gpt-4o-mini --no-defense --cases 20 --log results/runs.jsonl

# Vary jury size — does more jurors reduce variance?
python main.py --model gpt-4o-mini --jurors 1 --cases 20 --log results/runs.jsonl
python main.py --model gpt-4o-mini --jurors 5 --cases 20 --log results/runs.jsonl

# Vary debate depth — are more rounds worth the token cost?
python main.py --model gpt-4o-mini --max-rounds 2 --cases 20 --log results/runs.jsonl
python main.py --model gpt-4o-mini --max-rounds 6 --cases 20 --log results/runs.jsonl
```

All ablation flags are recorded in the JSONL log config so runs are self-documenting and reproducible.

| Ablation | What it tests | Flag |
|----------|--------------|------|
| No citation filter | Whether grounding constraints improve accuracy or just add latency | `--no-filter` |
| No defense | Whether adversarial structure outperforms single-annotator classification | `--no-defense` |
| Jury size | Variance reduction from ensemble voting (1, 3, 5 jurors) | `--jurors N` |
| Debate depth | Token cost vs. accuracy tradeoff at different round limits | `--max-rounds N` |


## Testing

```bash
# Run all tests
pytest test_agents.py test_graph.py -v

# Agent-only tests (mocked models, no API calls)
pytest test_agents.py -v

# Graph integration tests (filter logic, routing, end-to-end)
pytest test_graph.py -v
```


## Supported Models

| Provider | Models | Config |
|----------|--------|--------|
| Ollama (local) | `llama3:8b-instruct-q4_K_M`, `mistral:7b-instruct-q4_K_M`, `qwen2.5:14b-instruct-q4_K_M` | Automatic via `config.py` |
| OpenAI | `gpt-4o`, `gpt-4o-mini`, or any OpenAI chat model | Requires `OPENAI_API_KEY` in `.env` |

To add a new local model, add its name to the `local_models` set in `config.py`.


## Key Design Decisions

**Annotation framing over adversarial roles.** Early iterations used "Prosecutor" / "Defense Counsel" role names, which triggered safety refusals in smaller models (llama3:8b). Reframing as "content-analysis annotator" performing evidence extraction eliminated refusals without changing the adversarial structure.

**Few-shot examples in system prompts.** Small models pattern-match to examples more reliably than they follow abstract instructions. Each agent's system prompt includes a concrete example of the expected output format.

**Symmetric citation filter.** Both sides are held to the same grounding standard. The filter handles straight quotes, curly quotes, and guillemets, with a sliding-window fuzzy match for longer quotes to accommodate minor imprecisions from smaller models.

**Consecutive-failure retry budget.** Each agent gets one retry per turn if the citation filter rejects its argument. On a second consecutive failure the turn is forfeited — the argument is dropped from the transcript entirely so downstream agents never see ungrounded claims.

**Judge defaults to CLOSE.** Earlier versions defaulted to CONTINUE, causing debates to always run to max rounds. Flipping the default and requiring all three CONTINUE conditions to hold produces earlier closure when arguments stagnate.
