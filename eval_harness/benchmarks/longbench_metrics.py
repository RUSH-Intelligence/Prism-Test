"""Faithful port of the official LongBench metric suite.

Mirrors THUDM/LongBench ``metrics.py`` and the ``dataset2metric`` routing in
``eval.py`` so our numbers line up with the paper / leaderboard. LongBench scores
each task with a *task-specific* metric (token-F1, ROUGE-L, classification
accuracy, retrieval/count exact-match, code fuzzy-similarity) rather than one
uniform metric — using a single token-F1 everywhere (as an earlier revision did)
silently mis-scores summarization, classification, retrieval, count and code
tasks, and breaks Chinese tasks entirely (whitespace tokenization on CJK text).

Deviations from upstream:
  - ``code_sim_score`` calls ``fuzzywuzzy.fuzz.ratio`` (the upstream choice).
    With ``python-Levenshtein`` installed (declared in ``pyproject.toml``),
    fuzzywuzzy uses its fast C backend, matching the path most published
    LongBench numbers were measured on. If ``fuzzywuzzy`` is missing entirely,
    we fall back to a pure-Python ``difflib`` shim that produces the same
    score as fuzzywuzzy's own ``difflib`` fallback path (used when neither
    ``python-Levenshtein`` nor ``rapidfuzz`` is installed).
  - ``rouge`` (ROUGE-L) and ``jieba`` (CJK segmentation) are imported lazily so
    the framework still imports on nodes that only run the dependency-free
    benchmarks; a clear error is raised only when a ROUGE/Chinese task is scored
    without the library present.
"""
from __future__ import annotations

import difflib
import re
import string
from collections import Counter
from typing import Callable, Dict, List

try:
    from fuzzywuzzy import fuzz as _fuzz  # uses python-Levenshtein backend if present
    _HAS_FUZZ = True
except ImportError:  # pragma: no cover
    _fuzz = None
    _HAS_FUZZ = False


# --- normalization helpers (verbatim from upstream) -------------------------

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


_CN_PUNCTUATION = (
    "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』"
    "【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
)


def normalize_zh_answer(s: str) -> str:
    """Lower text and remove (ASCII + Chinese) punctuation and whitespace."""
    all_punctuation = set(string.punctuation + _CN_PUNCTUATION)
    no_punc = "".join(ch for ch in s.lower() if ch not in all_punctuation)
    return "".join(no_punc.split())


# --- token-overlap F1 (verbatim from upstream) ------------------------------

def _f1(prediction_tokens: List[str], ground_truth_tokens: List[str]) -> float:
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    return _f1(pred_tokens, gt_tokens)


def qa_f1_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    jieba = _require_jieba()
    pred_tokens = [normalize_zh_answer(t) for t in jieba.cut(prediction, cut_all=False)]
    gt_tokens = [normalize_zh_answer(t) for t in jieba.cut(ground_truth, cut_all=False)]
    pred_tokens = [t for t in pred_tokens if len(t) > 0]
    gt_tokens = [t for t in gt_tokens if len(t) > 0]
    if not pred_tokens or not gt_tokens:
        return 0.0
    return _f1(pred_tokens, gt_tokens)


# --- retrieval / count / classification / code -----------------------------

def count_score(prediction: str, ground_truth: str, **kwargs) -> float:
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def _retrieval_score(prediction: str, ground_truth: str, pattern: str) -> float:
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def retrieval_score(prediction: str, ground_truth: str, **kwargs) -> float:
    return _retrieval_score(prediction, ground_truth, r"Paragraph (\d+)")


def retrieval_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    return _retrieval_score(prediction, ground_truth, r"段落(\d+)")


def _fuzz_ratio(s1: str, s2: str) -> int:
    """``fuzzywuzzy.fuzz.ratio`` if available (matches upstream LongBench),
    else a difflib fallback equivalent to fuzzywuzzy's own pure-Python path."""
    if _HAS_FUZZ:
        return int(_fuzz.ratio(s1, s2))
    if not s1 and not s2:
        return 100
    matcher = difflib.SequenceMatcher(None, s1, s2)
    return int(round(100 * matcher.ratio()))


def code_sim_score(prediction: str, ground_truth: str, **kwargs) -> float:
    all_lines = prediction.lstrip("\n").split("\n")
    prediction = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction = line
            break
    return _fuzz_ratio(prediction, ground_truth) / 100


def classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
    all_classes = kwargs.get("all_classes") or []
    em_match_list = [c for c in all_classes if c in prediction]
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    return (1.0 / len(em_match_list)) if ground_truth in em_match_list else 0.0


# --- ROUGE-L (lazy dependency) ---------------------------------------------

def rouge_score(prediction: str, ground_truth: str, **kwargs) -> float:
    Rouge = _require_rouge()
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    jieba = _require_jieba()
    prediction = " ".join(jieba.cut(prediction, cut_all=False))
    ground_truth = " ".join(jieba.cut(ground_truth, cut_all=False))
    return rouge_score(prediction, ground_truth)


# --- lazy imports ----------------------------------------------------------

def _require_rouge():
    try:
        from rouge import Rouge
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise ImportError(
            "Scoring a LongBench ROUGE task (gov_report/qmsum/multi_news/samsum/"
            "dureader/vcsum) requires the 'rouge' package. Install it with "
            "`pip install rouge`."
        ) from exc
    return Rouge


def _require_jieba():
    try:
        import jieba
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise ImportError(
            "Scoring a Chinese LongBench task (multifieldqa_zh/dureader/vcsum/"
            "lsht/passage_retrieval_zh) requires 'jieba'. Install it with "
            "`pip install jieba`."
        ) from exc
    return jieba


# --- routing (verbatim from upstream eval.py) ------------------------------

DATASET2METRIC: Dict[str, Callable[..., float]] = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

# Datasets whose prediction is truncated to the first line before scoring
# (upstream eval.py / eval_e.py).
FIRST_LINE_TASKS = {"trec", "triviaqa", "samsum", "lsht"}


def base_task_name(task: str) -> str:
    """Strip the LongBench-E ``_e`` suffix so metric routing matches upstream."""
    return task[:-2] if task.endswith("_e") else task


def metric_for_task(task: str) -> Callable[..., float] | None:
    return DATASET2METRIC.get(base_task_name(task))
