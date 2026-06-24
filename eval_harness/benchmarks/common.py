from __future__ import annotations

import ast
import re
import string
from collections import Counter
from typing import Iterable, List


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = " ".join(text.split())
    return text


def parse_answers(value) -> List[str]:
    # Handle ndarray/Series-like values without a hard numpy dependency.
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            listed = value.tolist()
            if isinstance(listed, (list, tuple)):
                return [str(v) for v in listed]
        except Exception:
            pass

    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            # Handle numpy-style formatting: ['a' 'b' 'c'] (no commas).
            if re.search(r"'\s+'|\"\s+\"", s):
                quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
                recovered = [a or b for a, b in quoted if (a or b)]
                if recovered:
                    return recovered
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed]
            except Exception:
                quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
                recovered = [a or b for a, b in quoted if (a or b)]
                if recovered:
                    return recovered
        return [s]
    return [str(value)]


def exact_match_any(prediction: str, answers: Iterable[str]) -> float:
    p = normalize_text(prediction)
    refs = [normalize_text(a) for a in answers]
    return 1.0 if any(p == r for r in refs) else 0.0


def substring_match_any(prediction: str, answers: Iterable[str]) -> float:
    p = normalize_text(prediction)
    refs = [normalize_text(a) for a in answers]
    if not p:
        return 0.0
    return 1.0 if any((r and (r in p or p in r)) for r in refs) else 0.0


def token_f1_any(prediction: str, answers: Iterable[str]) -> float:
    p_toks = normalize_text(prediction).split()
    if not p_toks:
        return 0.0

    best = 0.0
    for ans in answers:
        a_toks = normalize_text(ans).split()
        if not a_toks:
            continue
        overlap = Counter(p_toks) & Counter(a_toks)
        same = sum(overlap.values())
        if same == 0:
            continue
        prec = same / len(p_toks)
        rec = same / len(a_toks)
        best = max(best, (2 * prec * rec) / (prec + rec))
    return best


def extract_option_letter(text: str) -> str:
    m = re.search(r"\b([A-D])\b", str(text).upper())
    return m.group(1) if m else ""


def extract_longbench_v2_answer(text: str) -> str:
    """Extract an A-D choice from a LongBench-v2 response.

    Mirrors the official THUDM/LongBench-v2 ``extract_answer``: strip markdown
    emphasis, then look for the steered final-answer template
    ``The correct answer is (X)`` / ``The correct answer is X``. The dataset's
    ``answer_prefix`` instructs the model to use exactly this format, so this is
    the faithful extraction. Falls back to a looser last/first standalone A-D
    match only when the template is absent, so a model that ignores the format
    instruction is not silently scored zero.
    """
    s = str(text).replace("*", "")
    m = re.search(r"The correct answer is \(([A-D])\)", s)
    if m:
        return m.group(1)
    m = re.search(r"The correct answer is ([A-D])", s)
    if m:
        return m.group(1)
    # Fallback: last standalone A-D mention (a final answer usually comes last).
    matches = re.findall(r"\b([A-D])\b", s.upper())
    return matches[-1] if matches else ""


def extract_int_0_999(text: str) -> str:
    t = str(text)
    boxed = re.findall(r"\\boxed\{([^}]*)\}", t)
    if boxed:
        nums = re.findall(r"\d+", boxed[-1])
        if nums:
            n = int(nums[-1])
            if 0 <= n <= 999:
                return str(n)

    nums = re.findall(r"\b\d+\b", t)
    for num in reversed(nums):
        n = int(num)
        if 0 <= n <= 999:
            return str(n)
    return ""
