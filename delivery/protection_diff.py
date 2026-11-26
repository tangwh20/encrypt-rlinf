"""Classify private source differences for an offline release without reporting source text."""

import ast
from collections import Counter
from dataclasses import dataclass
import difflib
import hashlib
import re


PRIORITY_RANK = {"none": 0, "low": 1, "review": 2, "high": 3}
TRAINING_TERMS = {"advantage", "advantages", "backward", "gradient", "gradients",
                  "loss", "optimizer", "policy", "reward", "rewards", "train", "training"}
SELECTION_TERMS = {"accept", "allow", "discard", "emit", "filter", "filtering", "gate",
                   "intervene", "keep", "reject", "retain", "select", "threshold"}
STATE_TERMS = {"buffer", "cache", "episode", "frames", "history", "pending", "state"}
MUTATORS = {"add", "append", "clear", "discard", "extend", "insert", "pop",
            "remove", "reset", "setdefault", "update"}


class _RemoveDocstrings(ast.NodeTransformer):
    def _scope(self, node):
        node.removed_doc = ast.get_docstring(node, clean=False)
        if node.removed_doc is not None:
            node.body = node.body[1:]
        return self.generic_visit(node)

    visit_Module = visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = _scope


class _Binding(ast.stmt):
    """A named child is compared separately from its parent scope."""

    _fields = ("name",)

    def __init__(self, name):
        self.name = name


@dataclass
class _Scope:
    kind: str
    line: int
    header: str
    body: list
    original_lines: bytes
    doc: str | None
    fingerprint: bytes | None
    binding_effects: bool


def _property_role(node):
    for decorator in getattr(node, "decorator_list", ()):
        if isinstance(decorator, ast.Name) and decorator.id == "property":
            return "getter"
        if isinstance(decorator, ast.Attribute) and decorator.attr in ("setter", "deleter", "getter"):
            return decorator.attr
    return None


def _fingerprint(node):
    return hashlib.sha256(ast.dump(node).encode("utf-8", errors="surrogatepass")).digest()


def _partition(data, path):
    tree = _RemoveDocstrings().visit(ast.parse(data, filename=path))
    lines = data.splitlines(keepends=True)
    scopes = {}
    occurrences = Counter()

    def collect(node, name, kind):
        # Under functions, closures remain part of their owner's implementation.
        # At class/module scope, even conditionally bound definitions get their own entry.
        key = None if kind == "module" else _fingerprint(node)
        bindings = bool(getattr(node, "decorator_list", ()) or getattr(node, "bases", ()) or
                        getattr(node, "keywords", ()))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = [*node.args.defaults, *node.args.kw_defaults]
            bindings |= any(isinstance(part, ast.Call) for default in defaults if default is not None
                            for part in ast.walk(default))
        body = node.body
        node.body = []
        header = ast.dump(node)
        if kind in ("module", "class"):
            class Children(ast.NodeTransformer):
                def definition(self, child):
                    role = _property_role(child)
                    qualified = child.name if kind == "module" else f"{name}.{child.name}"
                    if role:
                        qualified += f" [{role}]"
                    occurrences[qualified] += 1
                    if occurrences[qualified] > 1:
                        qualified += f" #{occurrences[qualified]}"
                    child_kind = "class" if isinstance(child, ast.ClassDef) else "function"
                    collect(child, qualified, role or child_kind)
                    return _Binding(qualified)

                visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = definition

            body = [Children().visit(statement) for statement in body]
        start = min([getattr(node, "lineno", 1)] + [
            item.lineno for item in getattr(node, "decorator_list", ())
        ])
        end = getattr(node, "end_lineno", len(lines))
        scopes[name] = _Scope(kind, start, header, body, b"".join(lines[start - 1:end]),
                              node.removed_doc, key, bindings)

    collect(tree, "<module>", "module")
    return scopes


def build_public_symbol_index(snapshot: dict[str, bytes]) -> dict:
    """Index complete normalized definitions, retaining up to two example locations."""
    index = {}
    for path in sorted(snapshot):
        if not path.endswith(".py"):
            continue
        try:
            scopes = _partition(snapshot[path], path)
        except (SyntaxError, ValueError, RecursionError):
            # When this path is directly compared, compare_module reports parsing failures.
            continue
        for name, scope in scopes.items():
            if scope.fingerprint is None:
                continue
            examples = index.setdefault(scope.fingerprint, [])
            if len(examples) < 2:
                examples.append({"file": path, "qualified_name": name, "line": scope.line})
    return index


def _shape(value, shared):
    """Preserve the order of common bindings relative to other executable statements."""
    if isinstance(value, list):
        return tuple(_shape(v, shared) for v in value
                     if not isinstance(v, _Binding) or v.name in shared)
    if isinstance(value, ast.AST):
        return (type(value).__name__, tuple(
            (key, _shape(v, shared)) for key, v in ast.iter_fields(value)
        ))
    # Keep 1, 1.0 and True distinct (Python compares their values equal).
    return (type(value).__name__, repr(value))


def _words(names):
    return set(re.findall(r"[a-z0-9]+", "_".join(names).lower()))


def _state_key(node):
    if isinstance(node, ast.Subscript):
        return _state_key(node.value)
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
            return f"{node.value.id}.{node.attr}"
        parent = _state_key(node.value)
        return f"{parent}.{node.attr}" if parent else None
    if isinstance(node, ast.Name) and _words([node.id]) & STATE_TERMS:
        return node.id
    return None


def _explain(name, nodes):
    """Signals are taken from new or changed statements, never from a directory alone."""
    expressions = [part for statement in nodes for part in ast.walk(statement)]
    names = {part.id if isinstance(part, ast.Name) else part.attr
             for part in expressions if isinstance(part, (ast.Name, ast.Attribute))}
    words = _words([name, *names])
    decisions = any(isinstance(part, (ast.If, ast.IfExp, ast.Match)) or
                    (isinstance(part, ast.comprehension) and part.ifs) for part in expressions)
    comparisons = any(isinstance(part, ast.Compare) for part in expressions)
    computes = any(isinstance(part, ast.BinOp) or
                   (isinstance(part, ast.Attribute) and part.attr in
                    {"backward", "clamp", "grad", "mean", "quantile", "std", "sum"})
                   for part in expressions)
    writes = set()
    tested_state = set()
    for part in expressions:
        if isinstance(part, (ast.Attribute, ast.Subscript)) and isinstance(part.ctx, (ast.Store, ast.Del)):
            writes.add(_state_key(part))
        if isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute):
            if part.func.attr in MUTATORS:
                writes.add(_state_key(part.func.value))
        if isinstance(part, (ast.If, ast.IfExp, ast.While)):
            tested_state.update(_state_key(n) for n in ast.walk(part.test))
    writes.discard(None)
    reasons = []
    if writes & tested_state:
        reasons.append("变更语句在条件中读取并写入同一状态字段，优先核验跨步决策。")
    if decisions and comparisons and words & SELECTION_TERMS:
        reasons.append("变更语句含筛选或干预相关标识、比较与分支，优先核验决策规则。")
    has_training_signal = (bool(_words([name]) & TRAINING_TERMS) or
                           bool(words & {"advantages", "backward", "gradient", "loss", "optimizer", "reward"}))
    if computes and has_training_signal and (comparisons or decisions or writes):
        reasons.append("训练或损失相关标识与数值运算及条件或状态变化同时出现。")
    if reasons:
        return "high", reasons
    if nodes and all(isinstance(n, (ast.Import, ast.ImportFrom, ast.Pass)) for n in nodes):
        return "low", ["差异集中在导入或占位语句。"]
    if len(nodes) == 1 and isinstance(nodes[0], ast.Return) and not decisions:
        if all(isinstance(part, (ast.Return, ast.Name, ast.Constant, ast.Attribute,
                                 ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.Load))
               for part in expressions):
            return "low", ["变更为简单取值、切片或直接返回；输入输出可能暴露其映射。"]
    return "review", ["当前交付含不同的可执行逻辑，需结合业务价值人工判定。"]


def _row(name, current, public, shared):
    scope = current or public
    row = {"qualified_name": name, "kind": scope.kind, "current_line": current.line if current else None,
           "public_line": public.line if public else None, "changes": []}
    if current is None:
        return dict(row, status="deleted", priority="none", reasons=["该公开定义已从当前模块删除。"])
    current_statements = [s for s in current.body if not isinstance(s, _Binding)]
    if public is None:
        if not current_statements:
            priority, reasons = "low", ["新建声明或命名空间，内部定义单独比较。"]
            if current.kind not in ("module", "class"):
                reasons = ["新增空函数声明。"]
        else:
            priority, reasons = _explain(name, current_statements)
        if priority == "low" and current.binding_effects:
            priority, reasons = "review", ["新增装饰器、类创建参数或计算默认值，需核验绑定行为。"]
        return dict(row, status="added", priority=priority, reasons=reasons)

    if current.header != public.header or current.kind != public.kind:
        row["changes"].append("declaration")
    old_statements = [s for s in public.body if not isinstance(s, _Binding) or s.name in shared]
    old_shapes = [_shape(s, shared) for s in old_statements]
    new_statements = [s for s in current.body if not isinstance(s, _Binding) or s.name in shared]
    new_shapes = [_shape(s, shared) for s in new_statements]
    if old_shapes == new_shapes:
        if row["changes"]:
            return dict(row, status="modified", priority="review",
                        reasons=["定义的签名、装饰器、基类或类型发生变化。"])
        if (current.doc != public.doc or (scope.kind not in ("module", "class") and
                                              current.original_lines != public.original_lines)):
            return dict(row, status="text_only", priority="low",
                        reasons=["定义内只有文档、注释或等价文本表示变化。"])
        return None
    row["changes"].append("statements")
    additions = []
    removals = []
    for tag, old_start, old_end, start, end in difflib.SequenceMatcher(
        None, old_shapes, new_shapes, autojunk=False
    ).get_opcodes():
        if tag in ("insert", "replace"):
            additions.extend(s for s in new_statements[start:end] if not isinstance(s, _Binding))
        if tag in ("delete", "replace"):
            removals.extend(s for s in old_statements[old_start:old_end] if not isinstance(s, _Binding))
    row["changed_current_lines"] = [statement.lineno for statement in additions]
    row["changed_public_lines"] = [statement.lineno for statement in removals]
    priority, reasons = _explain(name, additions)
    if not additions:
        reasons = ["当前执行路径因移除语句或调整声明次序而变化。"]
    return dict(row, status="modified", priority=priority, reasons=reasons)


def compare_module(path: str, current: bytes, public: bytes | None, *, public_symbols=None) -> dict:
    """Compare one shipped module to its counterpart in one local public snapshot."""
    if public == current:
        return {"status": "identical", "priority": "none", "reasons": ["与公开基线逐字节一致。"],
                "symbols": []}
    status = "added" if public is None else "modified"
    try:
        current_scopes = _partition(current, path)
        public_scopes = {} if public is None else _partition(public, path)
    except (SyntaxError, ValueError, RecursionError) as exc:
        line = getattr(exc, "lineno", None)
        return {"status": status, "priority": "review", "analysis_error": type(exc).__name__,
                "reasons": ["语法结构未能完整解析，需人工复核。"],
                "error_line": line, "symbols": []}
    shared = current_scopes.keys() & public_scopes.keys()
    rows = []
    for name in sorted(current_scopes.keys() | public_scopes.keys()):
        row = _row(name, current_scopes.get(name), public_scopes.get(name), shared)
        if row is not None:
            if public_symbols and row["status"] in ("added", "modified"):
                matches = public_symbols.get(current_scopes[name].fingerprint)
                if matches:
                    row.update(priority="low", public_matches=matches,
                               reasons=["去除文档与位置后，完整定义与公开快照中的定义相同。"])
            rows.append(row)
    if public is not None and all(row["status"] == "text_only" for row in rows):
        return {"status": "text_only", "priority": "low",
                "reasons": ["内容不同，忽略位置和文档后语法树等价。"], "symbols": rows}
    priority = max((row["priority"] for row in rows), key=PRIORITY_RANK.get, default="low")
    reasons = sorted({reason for row in rows if row["priority"] == priority for reason in row["reasons"]})
    if public is None and priority == "low" and any(row.get("public_matches") for row in rows):
        priority, reasons = "review", ["含公开已有定义的新模块，需核验模块组装及运行上下文。"]
    return {"status": status, "priority": priority, "reasons": reasons, "symbols": rows}
