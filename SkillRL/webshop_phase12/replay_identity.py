"""Bind training behavior while amending transport-only replay identity."""
import ast
import hashlib


def environment_behavior_fingerprint(text):
    tree = ast.parse(text)
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'ShopWorld':
            node.body = [n for n in node.body if not isinstance(n, ast.FunctionDef) or n.name != 'state_digest']
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
