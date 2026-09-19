"""
Structural features derived from the Abstract Syntax Tree of a code snippet.
AI-generated code tends to differ from human code in structural regularity:
more uniform branching, more consistent error handling patterns, fewer
"messy" real-world artifacts (dead code, inconsistent early returns, etc.)
"""
import ast
from collections import Counter
from typing import Dict

try:
    from radon.complexity import cc_visit
    from radon.metrics import h_visit
    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False


def safe_parse(code: str):
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def tree_depth(node, depth=0) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return depth
    return max(tree_depth(c, depth + 1) for c in children)


def node_type_distribution(tree) -> Dict[str, float]:
    counts = Counter()
    total = 0
    for node in ast.walk(tree):
        counts[type(node).__name__] += 1
        total += 1
    if total == 0:
        return {}
    # normalize to fractions so snippet length doesn't dominate
    return {f"ast_frac_{k}": v / total for k, v in counts.items()}


def extract_ast_features(code: str) -> Dict[str, float]:
    tree = safe_parse(code)
    if tree is None:
        return {
            "ast_parse_success": 0,
            "ast_depth": 0,
            "ast_n_nodes": 0,
            "ast_n_functions": 0,
            "ast_n_branches": 0,
            "ast_n_try_except": 0,
            "ast_n_comprehensions": 0,
            "ast_avg_function_args": 0,
            "ast_max_nesting": 0,
            "ast_cyclomatic_complexity": 0,
        }

    n_nodes = sum(1 for _ in ast.walk(tree))
    n_functions = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    n_branches = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.If, ast.For, ast.While)))
    n_try_except = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Try))
    n_comprehensions = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp))
    )

    func_arg_counts = [
        len(n.args.args) for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    avg_args = sum(func_arg_counts) / len(func_arg_counts) if func_arg_counts else 0

    cyclomatic = 0
    if RADON_AVAILABLE:
        try:
            blocks = cc_visit(code)
            cyclomatic = sum(b.complexity for b in blocks) / len(blocks) if blocks else 0
        except Exception:
            cyclomatic = 0

    features = {
        "ast_parse_success": 1,
        "ast_depth": tree_depth(tree),
        "ast_n_nodes": n_nodes,
        "ast_n_functions": n_functions,
        "ast_n_branches": n_branches,
        "ast_n_try_except": n_try_except,
        "ast_n_comprehensions": n_comprehensions,
        "ast_avg_function_args": avg_args,
        "ast_cyclomatic_complexity": cyclomatic,
    }
    features.update(node_type_distribution(tree))
    return features
