"""Read a pinned RLinf reference snapshot from local Git objects, without checkout."""

import io
import os
from pathlib import Path
import subprocess

from source_inventory import CALVIN_CONFIGS, SIMULATOR_DATA, SOURCE_TREES


DEFAULT_PUBLIC_COMMIT = "184ba07f0a5b3a09a4696a4e3c8aa4b3c720fa86"
DEFAULT_PUBLIC_URL = "https://github.com/RLinf/RLinf"


def _git(repository: Path, *arguments: str, data: bytes = b"",
         alternate_objects: Path | None = None) -> bytes:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        LC_ALL="C",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_ALLOW_PROTOCOL="",  # Also blocks promisor fetch on older Git.
        GIT_NO_LAZY_FETCH="1",
        GIT_GRAFT_FILE=os.devnull,
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_OPTIONAL_LOCKS="0",
        GIT_TERMINAL_PROMPT="0",
    )
    if alternate_objects is not None:
        # Git understands a C-quoted path here; octal bytes also preserve colons,
        # backslashes and non-UTF-8 directory names in this list-valued variable.
        path_bytes = os.fsencode(alternate_objects)
        env["GIT_ALTERNATE_OBJECT_DIRECTORIES"] = '"' + "".join(f"\\{b:03o}" for b in path_bytes) + '"'
    try:
        result = subprocess.run(
            ["git", "-c", "core.warnAmbiguousRefs=true", "-C", str(repository), *arguments],
            env=env, input=data, capture_output=True,
        )
    except OSError as exc:
        raise ValueError(f"Cannot run Git for the local public snapshot: {exc}") from exc
    message = result.stderr.decode("utf-8", "replace").strip()
    if result.returncode or "is ambiguous" in message:
        raise ValueError(f"Cannot read local public baseline in {repository}: {message}")
    return result.stdout


def _repository_root(repository: Path) -> Path:
    repository = repository.resolve()
    if not ((repository / ".git").exists()
            or ((repository / "HEAD").is_file() and (repository / "objects").is_dir())):
        raise ValueError(f"Expected a Git checkout or bare repository at {repository}")
    return repository


def _resolve_commit(repository: Path, revision: str) -> str:
    if not revision or revision.startswith("-") or "\0" in revision:
        raise ValueError(f"Invalid public baseline revision: {revision!r}")
    return _git(repository, "rev-parse", "--verify", "--end-of-options",
                f"{revision}^{{commit}}").decode("ascii").strip()


def select_public_merge_base(source: Path, public_repo: Path) -> tuple[str, str]:
    """Find a unique common ancestor of the two committed HEADs, using local objects."""
    source = _repository_root(source)
    public_repo = _repository_root(public_repo)
    common_dirs = [
        Path(os.fsdecode(_git(repo, "rev-parse", "--path-format=absolute",
                              "--git-common-dir").removesuffix(b"\n"))).resolve()
        for repo in (source, public_repo)
    ]
    hint = "Specify a known public commit with --public-ref."
    if common_dirs[0] == common_dirs[1]:
        raise ValueError(f"Automatic comparison needs an independent public repository. {hint}")
    try:
        source_head = _resolve_commit(source, "HEAD")
        public_head = _resolve_commit(public_repo, "HEAD")
        bases = _git(source, "merge-base", "--all", source_head, public_head,
                     alternate_objects=common_dirs[1] / "objects").decode("ascii").splitlines()
    except ValueError as exc:
        raise ValueError(f"No usable common ancestor in local history (missing, shallow, "
                         f"or unrelated repositories). {hint} {exc}") from exc
    if len(bases) != 1:
        raise ValueError(f"Expected a unique public merge base; found {len(bases)}. {hint}")
    # A shallow or partial public clone can omit a commit available only in the
    # source's store. Require the selected baseline itself to be readable there.
    try:
        selected = _resolve_commit(public_repo, bases[0])
    except ValueError as exc:
        raise ValueError(f"The public mirror lacks the selected commit. {hint}") from exc
    return selected, public_head


def read_public_snapshot(repository: Path, revision: str) -> tuple[str, dict[str, bytes]]:
    """Resolve one local commit and read its in-scope regular files as stored in Git."""
    repository = _repository_root(repository)
    commit = _resolve_commit(repository, revision)
    listing = _git(repository, "ls-tree", "-rz", "--full-tree", commit,
                   "--", *SOURCE_TREES, "LICENSE", "NOTICE")

    paths = {}
    for row in listing.split(b"\0"):
        if not row:
            continue
        attributes, raw_path = row.split(b"\t", 1)
        mode, kind, oid = attributes.split()
        name = os.fsdecode(raw_path)
        path = Path(name)
        if path not in CALVIN_CONFIGS and any(
            part.startswith(".") or part == "__pycache__" for part in path.parts
        ):
            continue
        if kind != b"blob" or mode not in (b"100644", b"100755"):
            raise ValueError(f"Unexpected symlink or non-regular Git entry in baseline: {name}")
        if (path.suffix in {".py", ".yaml", ".yml", ".sh"}
                or name in {"LICENSE", "NOTICE"} or path in SIMULATOR_DATA):
            paths[name] = oid

    objects = {}
    ids = dict.fromkeys(paths.values())
    request = b"".join(oid + b"\n" for oid in ids)
    with io.BytesIO(_git(repository, "cat-file", "--batch", data=request)) as stream:
        for oid in ids:
            header = stream.readline().split()
            if len(header) != 3 or header[:2] != [oid, b"blob"]:
                raise ValueError(f"Missing or invalid local baseline blob {oid.decode()}")
            size = int(header[2])
            content = stream.read(size)
            if len(content) != size or stream.read(1) != b"\n":
                raise ValueError(f"Truncated local baseline blob {oid.decode()}")
            objects[oid] = content

    return commit, {name: objects[oid] for name, oid in paths.items()}
