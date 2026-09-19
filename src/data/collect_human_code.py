"""
Collect human-written code samples from local git repositories (clone your
own list of permissively-licensed repos beforehand), extracting individual
function-level snippets so that granularity roughly matches the AI-generated
samples (which are also full function/solution level).

Usage:
    python -m src.data.collect_human_code --repos_dir data/raw/repos \
        --language python --out data/raw/human_code.jsonl \
        --max_snippets 1000
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import List, Dict


def extract_python_functions(file_path: Path) -> List[Dict]:
    """Parse a .py file and pull out standalone function definitions,
    filtering trivial/too-short/too-long ones to keep sample quality high."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []

    snippets = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                code = ast.get_source_segment(source, node)
            except Exception:
                continue
            if code is None:
                continue
            n_lines = code.count("\n") + 1
            if n_lines < 5 or n_lines > 120:
                continue  # keep snippet length comparable to typical LLM outputs
            snippet_id = hashlib.md5(code.encode()).hexdigest()[:12]
            snippets.append(
                {
                    "id": f"human_{snippet_id}",
                    "problem_id": None,
                    "language": "python",
                    "source_model": "human",
                    "label": 0,  # 0 = human-written
                    "code": code,
                    "source_file": str(file_path),
                }
            )
    return snippets


def collect_from_dir(repos_dir: str, language: str, max_snippets: int) -> List[Dict]:
    ext_map = {"python": "*.py", "java": "*.java", "cpp": "*.cpp", "javascript": "*.js"}
    pattern = ext_map.get(language, "*.py")
    all_snippets = []
    for file_path in Path(repos_dir).rglob(pattern):
        if language == "python":
            all_snippets.extend(extract_python_functions(file_path))
        if len(all_snippets) >= max_snippets:
            break
    return all_snippets[:max_snippets]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos_dir", required=True, help="Directory containing cloned repos")
    parser.add_argument("--language", default="python")
    parser.add_argument("--out", default="data/raw/human_code.jsonl")
    parser.add_argument("--max_snippets", type=int, default=1000)
    args = parser.parse_args()

    snippets = collect_from_dir(args.repos_dir, args.language, args.max_snippets)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for s in snippets:
            f.write(json.dumps(s) + "\n")
    print(f"Collected {len(snippets)} human snippets -> {args.out}")
