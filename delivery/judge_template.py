"""Remove non-executable comments from the staged judge script."""

import ast
import io
import tokenize
from pathlib import Path

JUDGE_MODULE = Path("rlinf/agents/rstar2/http_code_judge_tool.py")


def strip_judge_template_comments(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "code_template_setup"
    ]
    if len(assignments) != 1:
        raise ValueError(f"Review the judge template assignment in {path}")
    assignment = assignments[0]
    value = assignment.value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        raise ValueError(f"Review the judge template literal in {path}")

    template = value.value
    lines = template.split("\n")
    for token in tokenize.generate_tokens(io.StringIO(template).readline):
        if token.type == tokenize.COMMENT:
            row, start = token.start
            _, end = token.end
            lines[row - 1] = lines[row - 1][:start] + lines[row - 1][end:]
    stripped = "\n".join(lines)

    # Preserve the outer module's line numbers as well as the judge script's.
    start, end = value.lineno - 1, value.end_lineno - 1
    source_lines = source.split("\n")
    opening = source_lines[start].encode("utf-8")[: value.col_offset].decode("utf-8")
    closing = source_lines[end].encode("utf-8")[value.end_col_offset :].decode("utf-8")
    replacement = f"{opening}{stripped!r}" + "\n" * (end - start) + closing
    source_lines[start : end + 1] = [replacement]
    path.write_text("\n".join(source_lines), encoding="utf-8")
