"""
Stylistic / surface-level features that often distinguish AI from human code:
- Naming conventions (AI tends toward very consistent, descriptive names)
- Comment density and style (AI often over-comments in a templated way)
- Whitespace consistency (AI rarely leaves stray whitespace / tabs mixed with spaces)
- Token-level entropy (AI text often has lower "surprise")
"""
import math
import re
import tokenize
import io
from collections import Counter
from typing import Dict


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def naming_features(code: str) -> Dict[str, float]:
    identifiers = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code)
    if not identifiers:
        return {"name_avg_len": 0, "name_snake_case_ratio": 0, "name_single_char_ratio": 0}

    avg_len = sum(len(i) for i in identifiers) / len(identifiers)
    snake_case = sum(1 for i in identifiers if re.fullmatch(r"[a-z0-9_]+", i) and "_" in i)
    single_char = sum(1 for i in identifiers if len(i) == 1)

    return {
        "name_avg_len": avg_len,
        "name_snake_case_ratio": snake_case / len(identifiers),
        "name_single_char_ratio": single_char / len(identifiers),
    }


def comment_features(code: str) -> Dict[str, float]:
    lines = code.split("\n")
    n_lines = max(len(lines), 1)
    comment_lines = [l for l in lines if l.strip().startswith("#")]
    docstring_count = code.count('"""') + code.count("'''")

    avg_comment_len = (
        sum(len(l.strip("# ").strip()) for l in comment_lines) / len(comment_lines)
        if comment_lines else 0
    )

    return {
        "comment_density": len(comment_lines) / n_lines,
        "docstring_count": docstring_count / n_lines,
        "avg_comment_length": avg_comment_len,
    }


def whitespace_features(code: str) -> Dict[str, float]:
    lines = code.split("\n")
    n_lines = max(len(lines), 1)
    trailing_ws = sum(1 for l in lines if l != l.rstrip())
    mixed_indent = sum(1 for l in lines if l.startswith(" ") and "\t" in l)
    blank_lines = sum(1 for l in lines if l.strip() == "")
    line_lengths = [len(l) for l in lines]
    line_len_std = (
        (sum((x - sum(line_lengths) / n_lines) ** 2 for x in line_lengths) / n_lines) ** 0.5
        if n_lines > 0 else 0
    )

    return {
        "trailing_whitespace_ratio": trailing_ws / n_lines,
        "mixed_indent_ratio": mixed_indent / n_lines,
        "blank_line_ratio": blank_lines / n_lines,
        "line_length_std": line_len_std,
    }


def token_entropy_features(code: str) -> Dict[str, float]:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
        token_strings = [t.string for t in tokens if t.string.strip()]
    except (tokenize.TokenizeError, IndentationError, SyntaxError):
        token_strings = code.split()

    joined = " ".join(token_strings)
    return {
        "char_entropy": shannon_entropy(code),
        "token_entropy": shannon_entropy(joined),
        "n_tokens": len(token_strings),
        "unique_token_ratio": len(set(token_strings)) / len(token_strings) if token_strings else 0,
    }


def extract_style_features(code: str) -> Dict[str, float]:
    features = {}
    features.update(naming_features(code))
    features.update(comment_features(code))
    features.update(whitespace_features(code))
    features.update(token_entropy_features(code))
    return features
