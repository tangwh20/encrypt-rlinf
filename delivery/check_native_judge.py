"""Compare the native Code Judge with upstream using only fixed local submissions."""

import argparse
import asyncio
import base64
import ctypes
import ctypes.util
from dataclasses import asdict
import errno
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace


CLIENT = Path("rlinf/agents/rstar2/http_code_judge_tool.py")
LOADER = "from _rlinf_rstar2_sandbox import run; run("


def disable_network():
    """Deny sockets in this process and every child, including each judge process."""
    lib = ctypes.CDLL(ctypes.util.find_library("seccomp") or "libseccomp.so.2")
    lib.seccomp_init.argtypes = (ctypes.c_uint32,)
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = (ctypes.c_char_p,)
    lib.seccomp_rule_add.argtypes = (ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint)
    lib.seccomp_load.argtypes = (ctypes.c_void_p,)
    lib.seccomp_release.argtypes = (ctypes.c_void_p,)
    ctx = lib.seccomp_init(0x7FFF0000)
    if not ctx:
        raise RuntimeError("Could not prepare the network-denying seccomp filter.")
    try:
        for name in (b"socket", b"connect"):
            syscall = lib.seccomp_syscall_resolve_name(name)
            if syscall < 0 or lib.seccomp_rule_add(ctx, 0x00050000 | errno.EPERM, syscall, 0):
                raise RuntimeError(f"Could not deny {name!r}.")
        if lib.seccomp_load(ctx):
            raise RuntimeError("Could not install the network-denying seccomp filter.")
    finally:
        lib.seccomp_release(ctx)


def import_client(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_solution(sandbox, solution, data):
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(sandbox), "PYTHONNOUSERSITE": "1"}
    libc = Path(sys.base_prefix) / "lib"
    env["LD_LIBRARY_PATH"] = str(libc)
    result = subprocess.run([sys.executable, "-sBu", "-c", solution], cwd=sandbox,
                            env=env, text=True, encoding="utf-8", input=data,
                            capture_output=True, timeout=10, check=False)
    return {"exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


async def run_case(client, case, sandbox, native):
    name, args, expected = case
    calls = []

    async def send(url, request):
        assert url == "http://127.0.0.1:1/run/long-batch", url
        assert request.keys() == {"type", "submissions"} and request["type"] == "batch"
        submission, = request["submissions"]
        assert submission.keys() == {"type", "solution", "input"}
        assert submission["type"] == "python" and submission["input"] == args.get("input", "")
        solution = submission["solution"]
        if native:
            encoded = base64.b64encode(args.get("code", "").encode()).decode()
            assert solution == LOADER + repr(encoded) + ")"
        outcome = run_solution(sandbox, solution, submission["input"])
        calls.append((len(solution.encode()), outcome))
        return {"results": [{"run_success": True, "success": outcome["exit"] == 0,
                             "stdout": outcome["stdout"], "stderr": outcome["stderr"],
                             "reason": f"process exit {outcome['exit']}", "cost": 0.25}]}

    from rlinf.agents.tool_call.schema import ToolChannelRequest

    request = ToolChannelRequest("fixed-local", "execute", client.name, args)
    result = await client.execute(request, send)
    (solution_bytes, outcome), = calls
    assert expected == (outcome["exit"], outcome["stdout"]), (name, outcome)
    return solution_bytes, outcome, asdict(result)


async def compare(source, native, sandbox):
    sys.path.insert(0, str(native))
    path, = (native / CLIENT.parent).glob(CLIENT.stem + ".*.so")
    original = import_client("upstream_judge_client", source / CLIENT)
    compiled = import_client(CLIENT.stem, path)
    cfg = SimpleNamespace(tools=SimpleNamespace(
        codejudge=SimpleNamespace(host_addr="127.0.0.1", host_port=1)))
    original_tool = original.PythonTool(cfg)
    compiled_tool = compiled.PythonTool(cfg)
    assert original_tool.tool_schema() == compiled_tool.tool_schema()

    cases = [
        ("omitted code and input", {}, (0, "")),
        ("empty standard input", {"code": "import sys\nrepr(sys.stdin.read())", "input": ""}, (0, '"\'\'"\n')),
        ("input EOF exception", {"code": "input()", "input": ""}, (1, "")),
        ("standard input and unicode", {"code": "input()[::-1]", "input": "你好\n"}, (0, "'好你'\n")),
        ("standard output and error", {"code": "import sys\nprint('out')\nprint('err', file=sys.stderr)"}, (0, "out\n")),
        ("last multiline expression", {"code": "number = 41\n(\n number + 1\n)"}, (0, "42\n")),
        ("last expression returns None", {"code": "print('once')"}, (0, "once\n")),
        ("body exception", {"code": "print('before')\nraise ValueError('body')\npass"}, (1, "before\n")),
        ("final expression exception", {"code": "value = 1\n\nvalue / 0"}, (1, "")),
        ("syntax error", {"code": "x = ("}, (1, "")),
        ("explicit exit", {"code": "import sys\nsys.exit(7)"}, (7, "")),
        ("process state first request", {"code": "secret = 2\nsecret"}, (0, "2\n")),
        ("process state next request", {"code": "'secret' in globals()"}, (0, "False\n")),
    ]
    report = []
    for case in cases:
        old_size, old, old_response = await run_case(original_tool, case, sandbox, native=False)
        new_size, new, new_response = await run_case(compiled_tool, case, sandbox, native=True)
        assert old == new and old_response == new_response, (case[0], old, new)
        assert old_size > new_size, case[0]
        report.append({"case": case[0], "original_solution_bytes": old_size,
                       "native_solution_bytes": new_size, "process": new,
                       "client_response": new_response})
    submissions = ("session_only = 19\nprint(session_only)", "'session_only' in globals()")
    encoded = [base64.b64encode(code.encode()).decode() for code in submissions]
    upstream_batch = [
        original.code_template_setup + original.code_template_exec.format(code, "False") for code in encoded
    ]
    native_batch = [LOADER + repr(code) + ")" for code in encoded]

    def run_batch(requests):
        body = f"for submission in {requests!r}:\n    exec(submission, {{'__name__': '__main__'}})"
        return run_solution(sandbox, body, "")

    native_reused = run_batch(native_batch)
    assert run_batch(upstream_batch) == native_reused == {"exit": 0, "stdout": "19\nFalse\n", "stderr": ""}
    return report, native_reused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True,
                        help="directory containing the compiled client, schema, and sandbox modules")
    args = parser.parse_args()
    disable_network()
    from socket import AF_INET, socket

    try:
        socket(AF_INET)
    except PermissionError:
        pass
    else:
        raise RuntimeError("The local network-denying filter did not take effect.")
    native = args.native.resolve()
    sandbox_extension, = native.glob("_rlinf_rstar2_sandbox.*.so")
    with TemporaryDirectory(prefix="rlinf-fixed-judge-") as temp:
        site_packages = Path(temp) / "site-packages"
        site_packages.mkdir()
        shutil.copy2(sandbox_extension, site_packages)
        report, reused_process = asyncio.run(compare(args.source.resolve(), native, site_packages))
    compiled_client, = (native / CLIENT.parent).glob(CLIENT.stem + ".*.so")
    fingerprints = {}
    for module in (compiled_client, sandbox_extension):
        with module.open("rb") as stream:
            fingerprints[module.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    print(json.dumps({"network": "socket and connect denied by seccomp",
                      "process_per_submission": True, "native_sha256": fingerprints,
                      "reused_process_state_reset": reused_process, "cases": report},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
