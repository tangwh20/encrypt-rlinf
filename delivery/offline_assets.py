"""Package explicitly selected data files for the offline profile."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import struct


TEXT_TYPES = {".json", ".jsonl", ".yaml", ".yml", ".txt", ".csv", ".tsv"}
ASSET_TYPES = TEXT_TYPES | {".safetensors", ".parquet", ".png", ".jpg", ".jpeg", ".webp"}
PRIVATE_NAMES = re.compile(
    r"(?:credentials?|secrets?|tokens?|passwords?|id_rsa|id_ed25519|netrc|"
    r"authorization|api[_-]?keys?|private[_-]?keys?)(?:[._-]|$)", re.IGNORECASE
)
CONTROL = re.compile(rb"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
EXECUTABLE_TEXT = re.compile(
    rb"^\s*(?:#!|#\s*include\s*[<\"]|@echo\b|(?:from\s+[\w.]+\s+import|"
    rb"import\s+[\w.]+(?:\s+as\s+\w+)?\s*$)|(?:async\s+)?(?:def|class|function)\s+\w+\s*[:(\[]|"
    rb"(?:python\d*(?:\.\d+)?|bash|sh|eval|exec|curl|wget|chmod|sudo|echo)\s+[\w$'\x22/.-]|"
    rb"(?:if|for|while)\s+.+(?:;\s*(?:then|do)|:\s*)$|"
    rb"(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*\s*\([^\r\n]*[)\x22\x27]|[A-Za-z_]\w*\s*=[^=]|<\?php)",
    re.IGNORECASE,
)
PEM_KEY = re.compile(rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")
YAML_TAG = re.compile(rb"(?:^|\s)[!](?![\s])")
YAML_DATA = re.compile(rb"^(?:---|\.\.\.|-\s|[\[{]|(?:[^:{}\[\]]+):(?:\s|$))")
MAX_TEXT_RECORD = 8 << 20
MAX_DOCUMENT = 32 << 20
CHUNK = 1 << 20
TAIL = 64 << 10
DTYPE_SIZES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}


def safe_path(path):
    if not path.parts or any(
        part.startswith(".") or PRIVATE_NAMES.match(part) or "\\" in part or ":" in part
        or any(ord(character) < 32 for character in part)
        for part in path.parts
    ):
        raise ValueError(f"Hidden, credential or invalid offline asset path: {path}")


def asset_kind(path):
    suffix = path.suffix.casefold()
    if suffix not in ASSET_TYPES:
        raise ValueError(f"Unsupported offline asset format: {path}")
    return suffix


def asset_entries(directory):
    mode = directory.lstat().st_mode if directory.exists() or directory.is_symlink() else 0
    if not stat.S_ISDIR(mode):
        raise ValueError(f"Expected an offline asset directory without symlinks: {directory}")
    safe_path(Path(directory.name))

    def visit(path, relative):
        safe_path(relative)
        file_stat = path.lstat()
        if stat.S_ISDIR(file_stat.st_mode):
            yield Path("assets") / relative, path
            for child in sorted(path.iterdir()):
                yield from visit(child, relative / child.name)
        elif stat.S_ISREG(file_stat.st_mode) and file_stat.st_nlink == 1:
            asset_kind(relative)
            yield Path("assets") / relative, path
        else:
            raise ValueError(f"Offline asset is a link or special file: {path}")

    yield Path("assets"), directory
    for child in sorted(directory.iterdir()):
        yield from visit(child, Path(child.name))


def validate_asset_directory(directory, protected_roots):
    if directory.is_symlink():
        raise ValueError(f"Expected an offline asset directory without symlinks: {directory}")
    directory = directory.resolve()
    if any(directory == root or root.is_relative_to(directory) for root in protected_roots):
        raise ValueError("Choose a dedicated data directory for --offline-assets.")
    for _ in asset_entries(directory):
        pass
    return directory


class AssetReader:
    def __init__(self, stream, size, name, observe=None):
        self.stream = stream
        self.size = size
        self.name = name
        self.length = 0
        self.digest = hashlib.sha256()
        self.tail = b""
        self.observe = observe

    def counted(self, data):
        self.length += len(data)
        if self.length > self.size:
            raise ValueError(f"Invalid offline asset size: {self.name}")
        self.digest.update(data)
        if self.observe and data:
            self.observe(data)
        self.tail = (self.tail + data)[-32:]
        return data

    def read(self, size):
        data = self.counted(self.stream.read(size))
        if len(data) != size:
            raise ValueError(f"Truncated offline asset: {self.name}")
        return data

    def line(self):
        data = self.counted(self.stream.readline(MAX_TEXT_RECORD + 1))
        if len(data) > MAX_TEXT_RECORD or (not data and self.length < self.size):
            raise ValueError(f"Oversized or truncated offline text record: {self.name}")
        return data

    def drain(self):
        while self.length < self.size:
            self.read(min(CHUNK, self.size - self.length))


def json_data(data, name):
    try:
        return json.loads(data.decode("utf-8"), parse_constant=invalid_constant,
                          object_pairs_hook=unique_object)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"Invalid offline JSON data in {name}: {exc}") from exc


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON object key")
        value[key] = item
    return value


def invalid_constant(value):
    raise ValueError(f"Unsupported JSON constant {value}")


def text_check(data, name):
    if CONTROL.search(data) or PEM_KEY.search(data):
        raise ValueError(f"Binary or credential contents in offline data file: {name}")
    try:
        data.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"Offline text data must be UTF-8: {name}") from exc


def inspect_text_asset(reader, suffix, inspect_text):
    if suffix == ".json":
        if reader.size > MAX_DOCUMENT:
            raise ValueError(f"Offline JSON document exceeds {MAX_DOCUMENT} bytes; use JSONL: {reader.name}")
        data = reader.read(reader.size)
        text_check(data, reader.name)
        json_data(data, reader.name)
        if inspect_text:
            inspect_text(reader.name, data)
        return

    code_window = bytearray()
    has_yaml_structure = False
    while line := reader.line():
        text_check(line, reader.name)
        stripped = line.lstrip()
        if suffix == ".jsonl":
            if stripped:
                json_data(line, reader.name)
        elif EXECUTABLE_TEXT.match(line):
            raise ValueError(f"Executable text in offline data file: {reader.name}")
        elif suffix in {".yaml", ".yml"}:
            if YAML_TAG.search(line) or stripped.startswith(b"%"):
                raise ValueError(f"YAML tags or directives in offline data file: {reader.name}")
            has_yaml_structure = has_yaml_structure or bool(YAML_DATA.match(stripped))
        if inspect_text:
            code_window.extend(line)
            if len(code_window) >= CHUNK:
                inspect_text(reader.name, bytes(code_window))
                del code_window[:-TAIL]
    if suffix in {".yaml", ".yml"} and not has_yaml_structure:
        raise ValueError(f"Expected structured YAML data in {reader.name}")
    if inspect_text and code_window:
        inspect_text(reader.name, bytes(code_window))


def inspect_safetensors(reader, inspect_text):
    if reader.size < 10:
        raise ValueError(f"Invalid safetensors file: {reader.name}")
    header_size, = struct.unpack("<Q", reader.read(8))
    if not 2 <= header_size <= min(MAX_DOCUMENT, reader.size - 8):
        raise ValueError(f"Invalid safetensors header length: {reader.name}")
    header = reader.read(header_size)
    text_check(header, reader.name)
    content = json_data(header, reader.name)
    if not isinstance(content, dict):
        raise ValueError(f"Expected a safetensors header object: {reader.name}")
    tensors = []
    for key, value in content.items():
        if key == "__metadata__":
            if not isinstance(value, dict) or any(not isinstance(item, str) for item in value.values()):
                raise ValueError(f"Invalid safetensors metadata: {reader.name}")
            continue
        if not isinstance(value, dict) or not isinstance(value.get("dtype"), str):
            raise ValueError(f"Invalid safetensors tensor: {reader.name}")
        dtype_size = DTYPE_SIZES.get(value["dtype"])
        shape, offsets = value.get("shape"), value.get("data_offsets")
        if (dtype_size is None or not isinstance(shape, list) or not isinstance(offsets, list)
                or len(offsets) != 2 or any(type(part) is not int or part < 0 for part in shape + offsets)):
            raise ValueError(f"Invalid safetensors tensor shape or offsets: {reader.name}")
        length = dtype_size
        for dim in shape:
            length *= dim
        start, end = offsets
        if end - start != length:
            raise ValueError(f"Invalid safetensors tensor length: {reader.name}")
        tensors.append((start, end))
    expected_end = 0
    for start, end in sorted(tensors):
        if start != expected_end:
            raise ValueError(f"Invalid safetensors layout: {reader.name}")
        expected_end = end
    if reader.size != reader.length + expected_end:
        raise ValueError(f"Unexpected bytes after the safetensors tensors: {reader.name}")
    if inspect_text:
        inspect_text(reader.name, header)
    reader.drain()


def inspect_binary(reader, suffix):
    if suffix == ".parquet":
        valid_start = reader.size >= 12 and reader.read(4) == b"PAR1"
    elif suffix == ".png":
        valid_start = (reader.size >= 45 and
                       reader.read(16) == b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    elif suffix in {".jpg", ".jpeg"}:
        valid_start = reader.size >= 4 and reader.read(3) == b"\xff\xd8\xff"
    else:
        if reader.size < 20:
            raise ValueError(f"Invalid signature for offline asset format: {reader.name}")
        signature, length, tag, chunk = struct.unpack("<4sI4s4s", reader.read(16))
        valid_start = (signature == b"RIFF" and length + 8 == reader.size
                       and tag == b"WEBP" and chunk in {b"VP8 ", b"VP8L", b"VP8X"})
    if not valid_start:
        raise ValueError(f"Invalid signature for offline asset format: {reader.name}")
    reader.drain()
    if suffix == ".parquet":
        footer_size, = struct.unpack("<I", reader.tail[-8:-4])
        if reader.tail[-4:] != b"PAR1" or not 0 < footer_size <= reader.size - 12:
            raise ValueError(f"Invalid Parquet footer: {reader.name}")
    if suffix == ".png" and not reader.tail.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82"):
        raise ValueError(f"Invalid PNG end marker: {reader.name}")
    if suffix in {".jpg", ".jpeg"} and not reader.tail.endswith(b"\xff\xd9"):
        raise ValueError(f"Invalid JPEG end marker: {reader.name}")


def inspect_asset(stream, archive_name, size, inspect_text=None, observe=None):
    path = PurePosixPath(archive_name)
    suffix = asset_kind(path)
    reader = AssetReader(stream, size, archive_name, observe)
    if suffix in TEXT_TYPES:
        inspect_text_asset(reader, suffix, inspect_text)
    elif suffix == ".safetensors":
        inspect_safetensors(reader, inspect_text)
    else:
        inspect_binary(reader, suffix)
    return {"size": size, "sha256": reader.digest.hexdigest()}
