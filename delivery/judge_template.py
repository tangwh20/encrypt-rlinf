"""Prepare the RStar2 client for the source-free training distribution."""

import ast
from pathlib import Path

JUDGE_MODULE = Path("rlinf/agents/rstar2/http_code_judge_tool.py")
SANDBOX_MODULE = Path("_rlinf_rstar2_sandbox.py")
TEMPLATE_NAMES = ("code_template_setup", "code_template_exec")


def judge_templates(path: Path) -> dict[str, ast.Assign]:
    """Find the original literal assignments without importing the upstream module."""
    tree = ast.parse(path.read_bytes(), filename=str(path))
    templates = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in TEMPLATE_NAMES
        ):
            name = node.targets[0].id
            if name in templates or not (
                isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                raise ValueError(f"Review the judge template {name} in {path}")
            templates[name] = node
    if templates.keys() != set(TEMPLATE_NAMES):
        raise ValueError(f"Review the judge templates in {path}")
    return templates


def stage_native_judge(path: Path, sandbox: Path) -> None:
    """Compile the wrapper separately and send only a fixed loader and submission."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    templates = judge_templates(path)
    setup = templates["code_template_setup"].value.value
    # Upstream creates one executor each time its standalone script starts.
    # A sandbox may import this native module once and call it more than once.
    initialization = "persistent_executor = PersistentExecutor()"
    if setup.count(initialization) != 1 or not setup.rstrip().endswith(initialization):
        raise ValueError(f"Review the judge submission lifecycle in {path}")
    setup = setup.rstrip().removesuffix(initialization)
    invoke = templates["code_template_exec"].value.value.format("RLINF_USER_INPUT", "False")
    literal = '"RLINF_USER_INPUT"'
    if invoke.count(literal) != 1:
        raise ValueError(f"Review the sandbox submission argument in {path}")
    invoke = invoke.replace(literal, "encoded")
    sandbox.write_text(setup + "\n\ndef run(encoded):\n    " + initialization + "\n" +
                       "".join("    " + line + "\n" for line in invoke.strip().splitlines()),
                       encoding="utf-8")
    for node in templates.values():
        lines[node.lineno - 1 : node.end_lineno] = ["\n"] * (
            node.end_lineno - node.lineno + 1
        )
    source = "".join(lines)
    old_request = """        final_code = code_template_setup
        # TODO: add history code here
        final_code += code_template_exec.format(code_to_execute, "False")
"""
    new_request = ('        final_code = f"from _rlinf_rstar2_sandbox import run; '
                   'run({code_to_execute!r})"\n')
    if source.count(old_request) != 1:
        raise ValueError(f"Review the judge client conversion in {path}")
    source = source.replace(old_request, new_request)
    if any(
        isinstance(node, ast.Name) and node.id in TEMPLATE_NAMES
        for node in ast.walk(ast.parse(source, filename=str(path)))
    ):
        raise ValueError(f"Unexpected template reference in {path}")
    path.write_text(source, encoding="utf-8")
