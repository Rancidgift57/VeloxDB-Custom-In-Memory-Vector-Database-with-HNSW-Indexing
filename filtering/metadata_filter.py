"""
filtering/metadata_filter.py
Small expression DSL that compiles filter specs like:

    {"category": "finance", "status": "active"}                # implicit AND
    {"$and": [{"category": "finance"}, {"score": {"$gte": 10}}]}
    {"$or": [{"category": "finance"}, {"category": "tech"}]}

into a callable `metadata -> bool`, which is what HNSWIndex.search()'s
`filter_fn` expects. This is intentionally dependency-free (no query
parser) — it's evaluated once per candidate node during traversal so it
needs to stay cheap.
"""
from __future__ import annotations
from typing import Any, Callable, Dict

_OPS: Dict[str, Callable[[Any, Any], bool]] = {
    "$eq": lambda a, b: a == b,
    "$ne": lambda a, b: a != b,
    "$gt": lambda a, b: a is not None and a > b,
    "$gte": lambda a, b: a is not None and a >= b,
    "$lt": lambda a, b: a is not None and a < b,
    "$lte": lambda a, b: a is not None and a <= b,
    "$in": lambda a, b: a in b,
    "$nin": lambda a, b: a not in b,
}


def compile_filter(spec: Dict[str, Any]) -> Callable[[Dict[str, Any]], bool]:
    if not spec:
        return lambda meta: True

    def evaluate(meta: Dict[str, Any], node: Dict[str, Any]) -> bool:
        if "$and" in node:
            return all(evaluate(meta, sub) for sub in node["$and"])
        if "$or" in node:
            return any(evaluate(meta, sub) for sub in node["$or"])
        for field, cond in node.items():
            value = meta.get(field)
            if isinstance(cond, dict):
                for op, target in cond.items():
                    if op not in _OPS or not _OPS[op](value, target):
                        return False
            else:
                if value != cond:
                    return False
        return True

    return lambda meta: evaluate(meta, spec)
