"""Audit the packaged launcher and every primary RLinf payload member before publication."""

import argparse
import ast
import bz2
from collections import Counter, defaultdict, deque
import gzip
import hashlib
import importlib.util
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import re
import struct
import tarfile
import tempfile
import textwrap
import tokenize
import zipfile
import zlib

from sealed_modules import launcher_key, open_member
from source_inventory import SOURCE_TREES, inventory

FOOTER = struct.Struct("<8sQ32s")
OFFLINE_FOOTER = struct.Struct("<8sQQ32s")
MEMBER = struct.Struct("<cHIQ")
BLOCK = 1 << 20
MAX_MEMBER = 256 << 20
MAX_DECODED = 2 << 30
MAX_DEPTH = 12
SOURCE_SUFFIXES = {
    ".py", ".pyw", ".pyi", ".pyx", ".pxd", ".pxi", ".pyc", ".pyo", ".ipynb",
    ".c", ".cc", ".cpp", ".cxx", ".cu", ".h", ".hh", ".hpp", ".rs", ".go",
    ".sh", ".bash", ".zsh", ".js", ".ts", ".mjs", ".lua", ".ps1", ".bat", ".cmd",
    ".o", ".obj", ".dwo", ".dwp", ".pdb", ".debug", ".gcno", ".gcda",
}
# Exact known public dependency files with reviewed overlap against a named upstream
# adaptation. A changed public file must be reviewed again. Complete files and embedded
# private programs never use this exception.
PUBLIC_OVERLAP = Path(__file__).with_name("source_audit_public.json")
COMPRESSED_SUFFIXES = {".zip", ".whl", ".egg", ".tar", ".tgz", ".tbz", ".tbz2", ".txz",
                       ".gz", ".bz2", ".xz", ".lzma", ".zst", ".7z", ".rar"}
SIDECARS = ("README.md", "example.yaml", "install.sh")
LAUNCHER_NAME = "run"
LEGACY_LAUNCHER_NAME = "rlinf-train"
LAUNCHER_NAMES = (LAUNCHER_NAME, LEGACY_LAUNCHER_NAME)
PUBLIC_SCRIPTS = {
    "start.py", "app/launcher.py", "setup/finish_venv.py", "setup/finalize_runtime.py",
    "setup/check_environment.py", "setup/recipes/frankasim_ppo_mlp/build_runtime.sh",
    "setup/recipes/frankasim_ppo_mlp/build_toolchain.py",
    "setup/recipes/frankasim_ppo_mlp/finalize_runtime.py",
}
CODE_LINE = re.compile(rb"^(?:async\s+|)(?:def |class |from .+ import |import |return |raise |"
                       rb"for .+:|if .+:|while .+:|try:|except .*:)|^[\w.\[\]]+\s*(?:[:+*-]?=)[^=]")
READABLE = re.compile(rb"[\t\r\n\x20-\x7e\x80-\xf4]{20,}")
STREAM_HEADER = re.compile(rb"BZh[1-9]|\x1f\x8b\x08|\xfd7zXZ\x00|\x78[\x01\x5e\x9c\xda]")
FOREIGN_ARCHIVE = (b"\x28\xb5\x2f\xfd", b"7z\xbc\xaf\x27\x1c", b"Rar!\x1a\x07")
PRINTABLE = re.compile(rb"[\t\r\n\x20-\x7e\x80-\xf4]*\Z")
ZIP_MARKERS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
BYTECODE = re.compile(rb"(?:[\xa0-\xff][\x0b-\x11]|\x03\xf3|\xd1\xf2)\r\n")


class BoundedReader(io.RawIOBase):
    """Expose only one on-disk archive region and hash compressed bytes as they are read."""

    def __init__(self, stream, size):
        self.stream = stream
        self.remaining = size
        self.digest = hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        count = self.stream.readinto(memoryview(buffer)[:self.remaining])
        if not count and self.remaining:
            raise ValueError("Truncated release region")
        self.remaining -= count
        self.digest.update(memoryview(buffer)[:count])
        return count


def read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("Truncated release member")
    return data


def file_identity(stat):
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def hash_region(stream, size):
    digest = hashlib.sha256()
    while size:
        chunk = read_exact(stream, min(size, BLOCK))
        digest.update(chunk)
        size -= len(chunk)
    return digest.digest()


def valid_name(name):
    return (len(name) < 4096 and "\0" not in name
            and all(part and part not in {".", ".."} for part in name.split("/")))


def lines(data):
    for number, line in enumerate(data.splitlines(), 1):
        stripped = line.strip()
        if stripped and not stripped.startswith(b"#"):
            yield number, re.sub(rb"[ \t]+", b" ", stripped)


def windows(data, excluded=(), width=4):
    excluded = set(excluded)
    window = deque(maxlen=width)
    for number, line in lines(data):
        if number in excluded:
            window.clear()
            continue
        window.append((number, line))
        if len(window) == width:
            yield window[0][0], tuple(text for _, text in window)


def distinct_logic(fragment):
    return (len(set(fragment)) >= 3 and sum(map(len, fragment)) >= 64
            and sum(bool(CODE_LINE.match(line)) and not line.startswith((b"from ", b"import "))
                    for line in fragment) >= 2)


def template_literals(path, data):
    tree = ast.parse(data, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            raw = node.value.encode("utf-8", errors="surrogatepass") if isinstance(node.value, str) else node.value
            if len(raw) < 64 or sum(bool(CODE_LINE.match(line)) for _, line in lines(raw)) < 3:
                continue
            try:
                template = ast.parse(textwrap.dedent(raw.decode("utf-8")))
            except (SyntaxError, UnicodeError, ValueError):
                continue
            if any(isinstance(statement, (ast.Assign, ast.Import, ast.ImportFrom, ast.Try,
                                          ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef,
                                          ast.For, ast.If, ast.While, ast.With))
                   for statement in template.body):
                yield node.lineno, raw


class Reference:
    def __init__(self, source):
        modules, public = inventory(source)
        source_files = modules | {
            path.relative_to(source)
            for tree in SOURCE_TREES for path in (source / tree).rglob("*")
            if path.is_file() and not path.is_symlink()
            and path.suffix.lower() in SOURCE_SUFFIXES - {".pyc", ".pyo", ".o", ".obj"}
            and not any(part.startswith(".") for part in path.relative_to(source).parts)
        }
        self.complete = defaultdict(list)
        self.fragments = {width: defaultdict(dict) for width in (4, 8)}
        self.templates = []
        self.rstar2_literals = []
        self.source_files = len(source_files)
        for path in sorted(source_files - public):
            data = (source / path).read_bytes()
            # Empty/whitespace-only files do not identify an application's source.
            if not data.strip():
                continue
            self.complete[hashlib.sha256(data).digest()].append(str(path))
            excluded = set()
            if path.suffix == ".py":
                for token in tokenize.tokenize(io.BytesIO(data).readline):
                    if token.type == tokenize.STRING and token.start[0] < token.end[0]:
                        excluded.update(range(token.start[0], token.end[0] + 1))
                for number, template in template_literals(path, data):
                    self.templates.append(f"{path}:{number}")
                    if path.name == "http_code_judge_tool.py":
                        self.rstar2_literals.append((template, f"{path}:{number}"))
                        for marker in (b"class CodeExecutionError(", b"class PersistentExecutor:",
                                       b"    def execute_code("):
                            start = template.find(marker)
                            if start >= 0:
                                self.rstar2_literals.append((template[start:start + 256], f"{path}:{number}"))
                    self.complete[hashlib.sha256(template).digest()].append(f"{path}:{number}")
                    self.add_windows(template, path, number, is_template=True)
            self.add_windows(data, path, excluded=excluded)

    def add_windows(self, data, path, offset=1, *, excluded=(), is_template=False):
        widths = (4,) if is_template else (4, 8)
        for width in widths:
            for number, fragment in windows(data, excluded, width):
                if distinct_logic(fragment):
                    origin = (str(path), offset + number - 1, is_template)
                    self.fragments[width][fragment[0]][fragment] = origin

    def matches(self, data, *, offline=False):
        matches = {}
        for path in self.complete.get(hashlib.sha256(data).digest(), ()):
            matches[("complete protected source", path)] = None
        for match in READABLE.finditer(data):
            chunk = match.group()
            if b"\n" in chunk:
                self.match_fragments(chunk, matches, offline=offline)
            if b"\\n" in chunk:
                # Encoded strings can occur inside a file that also has physical line endings.
                for line in chunk.splitlines():
                    if line.count(b"\\n") >= 3:
                        decoded = line.replace(b"\\r\\n", b"\n").replace(b"\\n", b"\n")
                        decoded = decoded.replace(b"\\t", b"\t").replace(b'\\"', b'"')
                        self.match_fragments(decoded, matches, offline=offline)
        return matches

    def match_fragments(self, chunk, matches, *, offline):
        width = 8 if offline else 4
        index = self.fragments[width]
        template_index = self.fragments[4]
        window = deque(maxlen=width)
        for _, line in lines(chunk):
            window.append(line)
            fragment = tuple(window)
            if len(window) == width and (candidates := index.get(window[0])):
                if origin := candidates.get(fragment):
                    path, number, is_template = origin
                    kind = "embedded protected program" if is_template else "protected source fragment"
                    matches.setdefault((kind, path), number)
            if offline and len(window) >= 4 and (candidates := template_index.get(window[-4])):
                if (origin := candidates.get(fragment[-4:])) and origin[2]:
                    path, number, _ = origin
                    matches.setdefault(("embedded protected program", path), number)


def elf_sections(data):
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected little-endian ELF64")
    offset = struct.unpack_from("<Q", data, 40)[0]
    entry_size, count, names_index = struct.unpack_from("<HHH", data, 58)
    if not offset and not count:
        return
    if entry_size != 64 or offset + entry_size * count > len(data) or names_index >= count:
        raise ValueError("invalid ELF section table")
    names = offset + names_index * entry_size
    start, size = struct.unpack_from("<QQ", data, names + 24)
    if start + size > len(data):
        raise ValueError("invalid ELF section names")
    table = data[start:start + size]
    for index in range(count):
        name_index = struct.unpack_from("<I", data, offset + index * entry_size)[0]
        if name_index >= len(table):
            raise ValueError("invalid ELF section name")
        yield table[name_index:table.index(b"\0", name_index)]


def compressed_stream(data, offset):
    header = data[offset:offset + 6]
    if header.startswith(b"BZh"):
        kind, decoder = "bzip2", bz2.BZ2Decompressor()
    elif header.startswith(b"\xfd7zXZ\0"):
        kind, decoder = "xz", lzma.LZMADecompressor(memlimit=MAX_MEMBER)
    else:
        kind = "gzip" if header.startswith(b"\x1f\x8b\x08") else "zlib"
        decoder = zlib.decompressobj(wbits=31 if kind == "gzip" else 15)
    chunks = []
    total = 0
    remainder = memoryview(data)[offset:]
    try:
        for start in range(0, len(remainder), BLOCK):
            chunk = decoder.decompress(remainder[start:start + BLOCK], MAX_MEMBER + 1 - total)
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_MEMBER:
                raise ValueError("decoded stream exceeds audit limit")
            if decoder.eof:
                consumed = min(start + BLOCK, len(remainder)) - len(decoder.unused_data)
                return kind, b"".join(chunks), consumed
    except (OSError, EOFError, zlib.error, lzma.LZMAError):
        return None
    return None


def release_executable(directory):
    launchers = [directory / name for name in LAUNCHER_NAMES if os.path.lexists(directory / name)]
    if len(launchers) != 1:
        raise ValueError(f"Expected exactly one release launcher in {directory}")
    launcher = launchers[0]
    if launcher.is_symlink() or not launcher.is_file():
        raise ValueError(f"Expected a regular release launcher: {launcher}")
    return launcher


def member_kind(name):
    if name in PUBLIC_SCRIPTS:
        return "publisher startup/installer"
    if name.startswith("setup/requirements/"):
        return "upstream dependency recipes"
    if name.startswith(("setup/seed/lib/", "setup/seed/bin/", "setup/seed/include/")):
        return "public CPython/graphics runtime"
    if name.startswith("app/") and name.endswith(".so"):
        return "application native ELF"
    return "release resources and configuration"


class Audit:
    def __init__(self, source, trusted_inputs=None, *, member_observer=None):
        self.member_observer = member_observer
        self.reference = Reference(source)
        self.trusted_inputs = trusted_inputs
        self.findings = []
        self.counts = Counter()
        self.archives = Counter()
        self.readable_code = defaultdict(list)
        self.decoded_size = 0
        self.shared_public = set()
        self.public_overlap = json.loads(PUBLIC_OVERLAP.read_text(encoding="utf-8"))

    def inspect(self, name, data, group, *, depth=0, trusted=False):
        if depth > MAX_DEPTH:
            raise ValueError(f"Archive nesting exceeds audit limit at {name}")
        self.decoded_size += len(data)
        if self.decoded_size > MAX_DECODED:
            raise ValueError(f"Decoded payload exceeds audit limit at {name}")
        self.counts[group] += 1
        suffix = PurePosixPath(name).suffix.lower()
        bytecode = data.startswith(importlib.util.MAGIC_NUMBER) or BYTECODE.match(data[:4])
        shebang = data.startswith(b"#!")
        generated = (b"/* Generated by Cython" in data[:1024]
                     or (suffix == ".html" and b"Generated by Cython" in data[:4096]))
        if suffix in SOURCE_SUFFIXES or bytecode or shebang or generated:
            self.readable_code[group].append(name)
            if not trusted:
                self.findings.append((name, "source, bytecode or debug artifact", None))
        self.inspect_private(name, data, trusted=trusted)
        if suffix == ".so" and group == "application native ELF" and not data.startswith(b"\x7fELF"):
            self.findings.append((name, "invalid application shared object", None))
        if data.startswith(b"\x7fELF") and not trusted:
            try:
                sections = elf_sections(data)
                for section in sections:
                    if section.startswith((b".debug", b".zdebug", b".gnu_debug", b".stab")) or section == b".symtab":
                        self.findings.append((name, "ELF debug or static symbol table", section.decode("ascii")))
            except (ValueError, struct.error) as exc:
                raise ValueError(f"{name}: {exc}") from exc
        if data.startswith(ZIP_MARKERS):
            self.inspect_zip(name, data, group, depth, trusted)
        elif data[257:262] == b"ustar" or suffix == ".tar":
            self.inspect_tar(name, data, group, depth, trusted)
        elif data.startswith(b"!<arch>\n"):
            self.inspect_ar(name, data, group, depth, trusted)
        elif data.startswith(FOREIGN_ARCHIVE) or suffix in {".7z", ".rar", ".zst", ".lzma"}:
            raise ValueError(f"Unsupported encoded archive at {name}")
        elif STREAM_HEADER.match(data):
            self.inspect_stream(name, data, group, depth, trusted)
        elif suffix in COMPRESSED_SUFFIXES:
            raise ValueError(f"Unrecognized or corrupt archive at {name}")
        elif not PRINTABLE.fullmatch(data):
            # Cython's string pools occur inside a native ELF, with independent stream checksums.
            next_offset = 0
            for match in STREAM_HEADER.finditer(data):
                if match.start() < next_offset:
                    continue
                decoded = compressed_stream(data, match.start())
                if decoded:
                    kind, content, consumed = decoded
                    next_offset = match.start() + consumed
                    self.archives[f"embedded {kind}"] += 1
                    self.inspect(f"{name}!{kind}@{match.start():#x}", content, group,
                                 depth=depth + 1, trusted=trusted)

    def inspect_private(self, name, data, *, trusted=False, offline=False):
        overlap = None
        if trusted and name.startswith(("setup/seed/", "offline!/runtime/")):
            path = name.removeprefix("setup/seed/").removeprefix("offline!/runtime/")
            entry = self.public_overlap.get(path)
            if entry and hashlib.sha256(data).hexdigest() == entry[0]:
                overlap = entry[1]
        for (kind, source), line in self.reference.matches(data, offline=offline).items():
            located_source = source if line is None else f"{source}:{line}"
            if overlap and kind == "protected source fragment" and source in overlap:
                self.shared_public.add((name, located_source))
            else:
                self.findings.append((name, kind, located_source))

    def begin_offline_file(self, name):
        name = f"offline!/{name}"
        digest = hashlib.sha256()
        snippets = self.reference.rstar2_literals
        overlap = max((len(data) - 1 for data, _ in snippets), default=0)
        tail = b""
        found = set()

        def update(chunk):
            nonlocal tail
            digest.update(chunk)
            data = tail + chunk
            for snippet, source in snippets:
                if source not in found and snippet in data:
                    found.add(source)
            tail = data[-overlap:] if overlap else b""

        def finish():
            for source in sorted(found):
                self.findings.append((name, "streamed embedded RStar2 program", source))
            for source in self.reference.complete.get(digest.digest(), ()):
                self.findings.append((name, "complete protected source", source))
            self.counts["offline streamed file"] += 1

        return update, finish

    def inspect_stream(self, name, data, group, depth, trusted):
        remaining = data
        chunks = []
        size = 0
        while remaining:
            decoded = compressed_stream(remaining, 0)
            if not decoded:
                raise ValueError(f"Corrupt compressed member at {name}")
            kind, content, consumed = decoded
            self.archives[kind] += 1
            chunks.append(content)
            size += len(content)
            if size > MAX_MEMBER:
                raise ValueError(f"Decoded stream exceeds audit limit at {name}")
            remaining = remaining[consumed:]
        path = PurePosixPath(name)
        if path.suffix.lower() in {".tgz", ".tbz", ".tbz2", ".txz"}:
            child = f"{name}!{path.stem}.tar"
        elif path.suffix.lower() in {".gz", ".bz2", ".xz"}:
            child = f"{name}!{path.stem}"
        else:
            child = f"{name}!{kind}"
        self.inspect(child, b"".join(chunks), group, depth=depth + 1, trusted=trusted)

    def inspect_zip(self, name, data, group, depth, trusted):
        self.archives["zip"] += 1
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            seen = set()
            for member in archive.infolist():
                path = member.filename.rstrip("/")
                if not valid_name(path) or path in seen or member.file_size > MAX_MEMBER:
                    raise ValueError(f"Invalid ZIP member {name}!{path}")
                seen.add(path)
                if not member.is_dir():
                    nested_trust = trusted and not path.startswith(tuple(f"{tree}/" for tree in SOURCE_TREES))
                    self.inspect(f"{name}!/{path}", archive.read(member), group,
                                 depth=depth + 1, trusted=nested_trust)

    def inspect_tar(self, name, data, group, depth, trusted):
        self.archives["tar"] += 1
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
            for member in archive:
                path = member.name.removeprefix("./")
                if not valid_name(path) or member.size > MAX_MEMBER:
                    raise ValueError(f"Invalid tar member {name}!{path}")
                if member.isfile():
                    nested_trust = trusted and not path.startswith(tuple(f"{tree}/" for tree in SOURCE_TREES))
                    with archive.extractfile(member) as stream:
                        self.inspect(f"{name}!/{path}", read_exact(stream, member.size), group,
                                     depth=depth + 1, trusted=nested_trust)
                elif not (member.isdir() or member.issym() or member.islnk()):
                    raise ValueError(f"Unsupported tar member {name}!{path}")

    def inspect_ar(self, name, data, group, depth, trusted):
        self.archives["ar"] += 1
        stream = io.BytesIO(data)
        stream.seek(8)
        while header := stream.read(60):
            if len(header) != 60 or header[58:] != b"`\n":
                raise ValueError(f"Invalid ar member in {name}")
            path = header[:16].decode("ascii").strip()
            size = int(header[48:58])
            if not 0 <= size <= MAX_MEMBER:
                raise ValueError(f"Invalid ar member size in {name}")
            content = read_exact(stream, size)
            if size % 2:
                read_exact(stream, 1)
            if path not in {"/", "//", "/SYM64/"}:
                self.inspect(f"{name}!/{path.removesuffix('/')}", content, group,
                             depth=depth + 1, trusted=trusted)

    def binary(self, exe, offline_audit=None, *, sidecars=None):
        if sidecars is not None:
            if exe.absolute() != release_executable(sidecars).absolute():
                raise ValueError(f"Expected the audited executable in {sidecars}")
        elif exe.name in LAUNCHER_NAMES:
            release_executable(exe.parent)
        result = {}
        with exe.open("rb") as stream:
            original_stat = os.fstat(stream.fileno())
            size = stream.seek(0, 2)
            if size < FOOTER.size:
                raise ValueError("Missing release payload footer")
            stream.seek(size - FOOTER.size)
            magic, offset, digest = FOOTER.unpack(read_exact(stream, FOOTER.size))
            payload_end = size - FOOTER.size
            if magic != b"RLFRUN01" or not 0 < offset < payload_end:
                raise ValueError("Invalid release payload footer")
            launcher_end = offset
            if offset >= OFFLINE_FOOTER.size:
                stream.seek(offset - OFFLINE_FOOTER.size)
                magic, archive_start, archive_size, archive_digest = OFFLINE_FOOTER.unpack(
                    read_exact(stream, OFFLINE_FOOTER.size)
                )
                if magic == b"RLOFFL01":
                    if not 0 < archive_start < offset - OFFLINE_FOOTER.size or (
                        archive_start + archive_size != offset - OFFLINE_FOOTER.size
                    ):
                        raise ValueError("Invalid embedded offline archive bounds")
                    launcher_end = archive_start
                    stream.seek(archive_start)
                    if read_exact(stream, 3) != b"\x1f\x8b\x08":
                        raise ValueError("Embedded offline archive is not gzip")
                    stream.seek(archive_start)
                    if hash_region(stream, archive_size) != archive_digest:
                        raise ValueError("Embedded offline archive SHA-256 mismatch")
                    result["offline_archive"] = {
                        "offset": archive_start, "size": archive_size, "sha256": archive_digest.hex(),
                    }
                    if offline_audit is not None:
                        result["offline_archive"]["content_audit"] = offline_audit(
                            Path(f"/proc/self/fd/{stream.fileno()}"), archive_start, archive_size,
                            inspect_text=lambda name, data: self.inspect_private(
                                f"offline!/{name}", data, trusted=True,
                                offline=not name.startswith("assets/"),
                            ),
                            inspect_file=self.begin_offline_file,
                        )
            if launcher_end > MAX_MEMBER:
                raise ValueError("Launcher exceeds audit size limit")
            stream.seek(0)
            launcher = read_exact(stream, launcher_end)
            if not launcher.startswith(b"\x7fELF"):
                raise ValueError("Missing release launcher ELF")
            self.inspect(f"{exe.name}[launcher]", launcher, "publisher ELF launcher")
            stream.seek(offset)
            bounded = BoundedReader(stream, payload_end - offset)
            with gzip.GzipFile(fileobj=io.BufferedReader(bounded)) as payload:
                result["members"] = self.payload(payload, key=launcher_key(launcher))
            if bounded.remaining or bounded.digest.digest() != digest:
                raise ValueError("Release payload SHA-256 mismatch")
            final_stat = os.fstat(stream.fileno())
            if file_identity(original_stat) != file_identity(final_stat):
                raise ValueError("Release executable changed during source audit")
            if self.member_observer is not None:
                result["audited_file_identity"] = file_identity(final_stat)
            result["launcher_bytes"] = launcher_end
            result["application_payload"] = {
                "offset": offset, "size": payload_end - offset, "sha256": digest.hex(),
            }
        if sidecars is not None:
            result["release_sidecars"] = self.inspect_sidecars(sidecars)
        result.update({
            "reference_source_files": self.reference.source_files,
            "reference_complete_traceable_samples": sum(map(len, self.reference.complete.values())),
            "source_fragment_context_lines": {"application": 4, "offline_runtime": 8, "embedded_program": 4},
            "reference_embedded_programs": self.reference.templates,
            "inspected_objects_by_origin": dict(self.counts),
            "decoded_archives_and_streams": dict(self.archives),
            "readable_source_or_build_artifacts_by_origin": dict(self.readable_code),
            "shared_public_fragments": [
                dict(member=member, source=source) for member, source in sorted(self.shared_public)
            ],
            "findings": [dict(member=member, issue=kind, source=source)
                         for member, kind, source in self.findings],
        })
        return result

    def inspect_sidecars(self, directory):
        result = {}
        for name in SIDECARS:
            path = directory / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_MEMBER:
                raise ValueError(f"Invalid publication sidecar: {path}")
            data = path.read_bytes()
            installer = name == "install.sh"
            group = "publisher startup/installer" if installer else "release resources and configuration"
            self.inspect(f"sidecar!/{name}", data, group, trusted=installer)
            result[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return result

    def payload(self, payload, *, key=None):
        encrypted = {}
        seen = set()
        kinds = Counter()
        directories = []
        groups = Counter()
        while (kind := read_exact(payload, 1)) != b"\0":
            _, _, name_size, size = MEMBER.unpack(kind + read_exact(payload, MEMBER.size - 1))
            if not 0 < name_size < 4096 or size > MAX_MEMBER:
                raise ValueError("Invalid release member length")
            name = read_exact(payload, name_size).decode("utf-8", errors="surrogateescape")
            if not valid_name(name) or name in seen:
                raise ValueError(f"Invalid release member name {name!r}")
            seen.add(name)
            data = read_exact(payload, size)
            if kind == b"E":
                digest = hashlib.sha256(data).hexdigest()
                data = open_member(key, name, data)
                encrypted[name] = {"ciphertext_sha256": digest, "sha256": hashlib.sha256(data).hexdigest()}
                kind = b"F"
            kinds[kind.decode("ascii")] += 1
            if kind == b"F":
                if self.member_observer is not None:
                    self.member_observer(name, data)
                group = member_kind(name)
                if group == "release resources and configuration" and data.startswith(b"\x7fELF"):
                    group = "publisher native helper"
                groups[group] += 1
                trusted = group in {"publisher startup/installer", "upstream dependency recipes",
                                    "public CPython/graphics runtime"}
                if trusted and self.trusted_inputs is not None:
                    expected = self.trusted_inputs.get(name)
                    trusted = expected is not None and hashlib.sha256(data).digest() == expected
                    if not trusted:
                        self.findings.append((name, "unexpected public input or hash mismatch", None))
                self.inspect(name, data, group, trusted=trusted)
            elif kind == b"L":
                if not data or b"\0" in data or data.startswith(b"/"):
                    raise ValueError(f"Invalid symlink target at {name}")
            elif kind != b"D" or size:
                raise ValueError(f"Invalid release member type for {name}")
            elif self.member_observer is not None:
                directories.append(name)
        if payload.read(1):
            raise ValueError("Unexpected data after release member terminator")
        result = {"types": dict(kinds), "regular_files_by_origin": dict(groups), "encrypted_native_modules": encrypted}
        if self.member_observer is not None:
            result["directories"] = directories
        return result


def public_input_digests(records):
    public = {"publisher startup/installer", "upstream dependency recipes",
              "public CPython/graphics runtime"}
    result = {}
    for kind, name, path in records:
        if kind == b"F" and member_kind(str(name)) in public:
            with path.open("rb") as stream:
                result[str(name)] = hash_region(stream, path.stat().st_size)
    return result


def audit_release(exe, source, *, trusted_inputs=None, offline_audit=None, sidecars=None,
                  report_path=None, member_observer=None):
    audit = Audit(source, trusted_inputs, member_observer=member_observer)
    result = audit.binary(exe, offline_audit, sidecars=sidecars)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".source-audit-", dir=report_path.parent) as tmp:
            draft = Path(tmp) / report_path.name
            fd = os.open(draft, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(json.dumps(result, ensure_ascii=True, indent=2) + "\n")
            draft.replace(report_path)
    if findings := result["findings"]:
        samples = "; ".join(
            f"{finding['member']}: {finding['issue']} ({finding['source'] or 'member'})"
            for finding in findings[:10]
        )
        raise ValueError(f"Source audit rejected {len(findings)} finding(s): {samples}")
    return result


def main():
    from offline_runtime import audit_offline_runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--sidecars", type=Path, help="directory containing the three published sidecar files")
    args = parser.parse_args()
    result = Audit(args.source).binary(args.elf, audit_offline_runtime, sidecars=args.sidecars)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return bool(result["findings"])


if __name__ == "__main__":
    raise SystemExit(main())
