"""
Generate labeled AI-written code samples by prompting one or more LLM APIs
with coding problems, and saving the resulting solutions with metadata
(model name, problem id, language) for later use as the positive class.

Usage:
    python -m src.data.generate_ai_code --problems data/raw/problems.json \
        --models claude gpt4 --languages python --out data/raw/ai_code.jsonl
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict

CODE_PROMPT_TEMPLATE = """Write a {language} solution to the following problem.
Return ONLY the code, no explanations, no markdown fences.

Problem:
{problem}
"""


def call_anthropic(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def call_openai(prompt: str, model: str = "gpt-4o") -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


MODEL_DISPATCH = {
    "claude": call_anthropic,
    "gpt4": call_openai,
}


def clean_code_block(text: str) -> str:
    """Strip markdown fences if the model added them anyway."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def generate_dataset(
    problems: List[Dict],
    models: List[str],
    languages: List[str],
    out_path: str,
    sleep_between_calls: float = 1.0,
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_path, "w") as f:
        for problem in problems:
            for language in languages:
                prompt = CODE_PROMPT_TEMPLATE.format(
                    language=language, problem=problem["statement"]
                )
                for model_key in models:
                    fn = MODEL_DISPATCH.get(model_key)
                    if fn is None:
                        continue
                    try:
                        raw = fn(prompt)
                        code = clean_code_block(raw)
                        record = {
                            "id": f"{problem['id']}_{language}_{model_key}",
                            "problem_id": problem["id"],
                            "language": language,
                            "source_model": model_key,
                            "label": 1,  # 1 = AI-generated
                            "code": code,
                        }
                        f.write(json.dumps(record) + "\n")
                        n_written += 1
                    except Exception as e:
                        print(f"[warn] failed on {problem['id']}/{model_key}: {e}")
                    time.sleep(sleep_between_calls)
    print(f"Wrote {n_written} AI-generated samples to {out_path}")


def load_problems(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", required=True, help="JSON file: list of {id, statement}")
    parser.add_argument("--models", nargs="+", default=["claude"], choices=list(MODEL_DISPATCH.keys()))
    parser.add_argument("--languages", nargs="+", default=["python"])
    parser.add_argument("--out", default="data/raw/ai_code.jsonl")
    args = parser.parse_args()

    problems = load_problems(args.problems)
    generate_dataset(problems, args.models, args.languages, args.out)
