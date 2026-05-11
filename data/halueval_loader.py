import json
from urllib.parse import quote, urlencode
from urllib.request import urlopen


_DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
_REPO = "pminervini/HaluEval"
_DEFAULT_SPLIT = "data"  # HaluEval ships each subset under a single "data" split.
_PAGE_SIZE = 100  # datasets-server caps `length` at 100.


def _fetch_page(subset: str, split: str, offset: int, length: int) -> list[dict]:
    """Fetch one page of rows from the HuggingFace datasets-server REST API."""
    qs = urlencode({
        "dataset": _REPO,
        "config":  subset,
        "split":   split,
        "offset":  offset,
        "length":  length,
    }, quote_via=quote)
    url = f"{_DATASETS_SERVER}?{qs}"
    with urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read())
    return [item["row"] for item in payload.get("rows", [])]


def iter_halueval(subset: str = "summarization_samples", split: str = _DEFAULT_SPLIT):
    """Yield rows from a HaluEval subset by paging the datasets-server API."""
    offset = 0
    while True:
        page = _fetch_page(subset, split, offset, _PAGE_SIZE)
        if not page:
            return
        for row in page:
            yield row
        offset += len(page)
        if len(page) < _PAGE_SIZE:
            return


def get_halueval_cases(n: int, subset: str = "summarization_samples") -> list[dict]:
    """
    Return up to `n` cases shaped for the courtroom pipeline:
      - 'response' (str): the summary under review
      - 'prompt'   (str): instruction + source document
      - 'label'    (str): 'hallucinated' if hallucinated, else 'faithful'
                          (matches the SUMMARIZATION_FAITHFULNESS task labels)

    Class-balanced: roughly n/2 hallucinated and n/2 faithful summaries,
    in the order they appear in the dataset.
    """
    half = n // 2
    need_hallucinated = n - half
    need_faithful = half

    hallucinated, faithful = [], []
    for row in iter_halueval(subset):
        is_hallucinated = str(row["hallucination"]).strip().lower() == "yes"
        if is_hallucinated and len(hallucinated) < need_hallucinated:
            hallucinated.append(row)
        elif not is_hallucinated and len(faithful) < need_faithful:
            faithful.append(row)
        if len(hallucinated) >= need_hallucinated and len(faithful) >= need_faithful:
            break

    cases = []
    for row in hallucinated + faithful:
        cases.append({
            "response": row["summary"],
            "prompt":   f"Summarize the following document:\n\n{row['document']}",
            "label":    "hallucinated" if str(row["hallucination"]).strip().lower() == "yes" else "faithful",
        })
    return cases
