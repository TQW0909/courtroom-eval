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
├── filters/
│   └── citation_filter.py   # Verbatim-quote grounding check with retry/forfeit logic
│
├── graph/
│   └── courtroom_graph.py   # LangGraph state definition and graph wiring
│
├── data/
│   └── jbb_loader.py        # HuggingFace dataset loader for JBB-Behaviors
│
├── utils/
│   └── pretty_print.py      # Rich terminal output (panels, tables, color-coded verdicts)
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


## Usage

### Basic run

```bash
# Run 10 cases with GPT-4o-mini (default)
python main.py

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
| `--jury-model` | same as `--model` | Separate model for the 3 jurors |
| `--max-rounds` | `4` | Maximum debate rounds before forced close |
| `--cases` | `10` | Number of cases to evaluate |
| `--split` | `test` | JBB dataset split (`train` or `test`) |
| `--verbose` | off | Print full trial transcript for each case |

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
