"""Pack a finalized task prefix and the GPU library used by its pinned SAPIEN."""

import io
import io
import json
import os
import posixpath
from pathlib import Path, PurePosixPath
import stat
import tarfile
import tempfile
import zipfile
import zlib

from offline_assets import TEXT_TYPES, asset_entries, inspect_asset, safe_path as safe_asset_path


PREFIX_ROOTS = ("BUILD", "bin", "include", "lib", "lib64", "share", "toolchain")
IGNORED = {".git", ".hg", ".svn", ".cache", "__pycache__", ".ssh", ".aws", ".netrc",
           ".env", ".config", ".pypirc", ".npmrc", ".gitconfig"}
TEXT_LIMIT = 8 << 20
ZIP_COMPRESSED_LIMIT = 256 << 20
ZIP_EXPANDED_LIMIT = 512 << 20
ZIP_ENTRY_LIMIT = 128 << 20
ZIP_SPOOL_THRESHOLD = 8 << 20
ZIP_MARKERS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
ZIP_SUFFIXES = {".egg", ".jar", ".pyz", ".whl", ".zip"}
PROTECTED_PACKAGES = {"rlinf", "examples", "evaluations", "toolkits"}
PHYSX_LIBRARY = Path(".sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so")


def protected_package(name):
    parts = name.casefold().replace("-", ".").replace("_", ".").split(".")
    return parts[0] in PROTECTED_PACKAGES


def runtime_entries(runtime):
    """Yield the relocatable package tree without checkout or cache metadata."""
    for name in PREFIX_ROOTS:
        path = runtime / name
        if not path.exists() and not path.is_symlink():
            continue
        yield from runtime_tree(runtime, path)


def runtime_tree(runtime, path):
    rel = path.relative_to(runtime)
    if path.name in IGNORED or path.suffix in {".pyc", ".pyo"}:
        return
    if path.suffix == ".egg-link" or path.name.startswith("__editable__"):
        raise ValueError(f"Offline runtime contains an editable install: {rel}")
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        target = path.resolve(strict=True)
        if os.path.isabs(path.readlink()) or not target.is_relative_to(runtime):
            raise ValueError(f"Offline runtime contains a nonportable link: {rel}")
        target_parts = target.relative_to(runtime).parts
        if (not target_parts or target_parts[0] not in PREFIX_ROOTS
                or IGNORED.intersection(target_parts)):
            raise ValueError(f"Offline runtime contains a nonportable link: {rel}")
    elif not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        raise ValueError(f"Unsupported offline runtime member: {rel}")
    if path.name == "direct_url.json" and path.parent.name.endswith(".dist-info"):
        if json.loads(path.read_text()).get("dir_info", {}).get("editable"):
            raise ValueError(f"Offline runtime contains an editable install: {rel}")
        # Wheel provenance is unnecessary at runtime and can contain private URLs.
        return
    if path.parent.name == "site-packages":
        if protected_package(path.name):
            raise ValueError(f"Build the offline runtime without a separate RLinf install: {rel}")
        if path.suffix == ".pth":
            for line in path.read_text().splitlines():
                location = line.strip()
                if location and not location.startswith(("#", "import ", "import\t")):
                    target = (path.parent / location).resolve()
                    if Path(location).is_absolute() or not target.is_relative_to(runtime):
                        raise ValueError(f"Offline runtime contains a nonportable import path: {rel}")
    yield Path("runtime") / rel, path
    if stat.S_ISDIR(mode):
        for child in sorted(path.iterdir()):
            yield from runtime_tree(runtime, child)


def validate_offline_prefix(prefix):
    runtime = prefix.resolve()
    python = runtime / "bin/python"
    if (not python.is_file() or not os.access(python, os.X_OK)
            or (runtime / "pyvenv.cfg").exists()
            or not (runtime / "lib/python3.11/site-packages").is_dir()):
        raise ValueError(f"Expected a standalone CPython 3.11 prefix: {prefix}")
    return runtime


def validate_offline_inputs(prefix, home):
    runtime = validate_offline_prefix(prefix)
    assets = home.resolve() if home else runtime.parent / "home"
    physx = assets / PHYSX_LIBRARY
    if not physx.is_file() or physx.resolve() != physx:
        raise ValueError(f"Missing pinned SAPIEN PhysX GPU library: {physx}")
    with physx.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError(f"Expected the pinned SAPIEN PhysX GPU shared library: {physx}")
    return runtime, assets


def pack_offline_runtime(runtime, home, destination, *, assets=None):
    print(f"Packing the offline task runtime from {runtime}...", flush=True)
    # Generate one compressed archive directly from the source prefix.
    with tarfile.open(destination, "w:gz", compresslevel=1) as archive:
        def add(path, name):
            info = archive.gettarinfo(str(path), str(name))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode &= 0o755
            if name.parts[0] == "assets" and info.isreg():
                info.mode &= 0o644
            if name.parts[0] == "assets":
                if not path.resolve().is_relative_to(assets) or not (info.isreg() or info.isdir()):
                    raise ValueError(f"Offline asset changed while packing: {path}")
                if info.isreg():
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    with os.fdopen(fd, "rb") as stream:
                        opened = os.fstat(stream.fileno())
                        if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                                or opened.st_size != info.size):
                            raise ValueError(f"Offline asset changed while packing: {path}")
                        archive.addfile(info, stream)
                    return
            if info.isreg():
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
            else:
                archive.addfile(info)

        add(runtime, Path("runtime"))
        for name, path in runtime_entries(runtime):
            add(path, name)
        for rel in (*reversed(PHYSX_LIBRARY.parents), PHYSX_LIBRARY):
            add(home / rel, Path("home") / rel)
        if assets is not None:
            for name, path in asset_entries(assets):
                add(path, name)
    return destination


class ArchiveRegion:
    """Read only the supplied compressed range of the release executable."""

    def __init__(self, stream, length):
        self.stream = stream
        self.remaining = length

    def read(self, size):
        data = self.stream.read(min(size, self.remaining))
        self.remaining -= len(data)
        return data


class SingleGzipStream:
    """Decode one complete gzip member and reject any bytes after its trailer."""

    def __init__(self, source):
        self.source = source
        self.decoder = zlib.decompressobj(wbits=31)
        self.unread = b""

    def read(self, size):
        chunks = []
        remaining = size
        while remaining and not self.decoder.eof:
            data = self.unread or self.source.read(1 << 16)
            if not data:
                raise ValueError("Truncated offline gzip stream")
            decoded = self.decoder.decompress(data, remaining)
            self.unread = self.decoder.unconsumed_tail
            chunks.append(decoded)
            remaining -= len(decoded)
        if self.decoder.eof and (self.decoder.unused_data or self.source.read(1)):
            raise ValueError("Extra data after the offline gzip stream")
        return b"".join(chunks)


class StrictTarHeader(tarfile.TarInfo):
    @classmethod
    def fromtarfile(cls, archive):
        try:
            return super().fromtarfile(archive)
        except tarfile.EOFHeaderError:
            raise
        except tarfile.HeaderError as exc:
            raise ValueError(f"Invalid offline tar header: {exc}") from exc


def read_member(stream, size, name, inspect_file, *, capture=False, head=b"", spool=None):
    """Consume every logical byte while retaining only small, relevant contents."""
    observe, finish = inspect_file(name) if inspect_file else (None, None)
    captured = bytearray(head) if capture else None
    remaining = size - len(head)
    if head:
        if observe:
            observe(head)
        if spool is not None:
            spool.write(head)
    while remaining:
        data = stream.read(min(remaining, 1 << 20))
        if not data:
            raise ValueError(f"Truncated offline runtime member: {name}")
        remaining -= len(data)
        if capture:
            captured.extend(data)
        if spool is not None:
            spool.write(data)
        if observe:
            observe(data)
    if finish:
        finish()
    return bytes(captured) if capture else b""

def valid_zip_path(path, filename):
    relative = filename.removesuffix("/")
    if (str(path) != relative or path.is_absolute() or ".." in path.parts
            or not path.parts or "\\" in relative or ":" in relative
            or any(ord(character) < 32 for character in relative)):
        return False
    # Common package names like "examples" can be resources in vendor trees.
    # Treat them as protected only at actual Python package roots.
    for index, part in enumerate(path.parts):
        if not protected_package(part):
            continue
        package = part.casefold().replace("-", ".").replace("_", ".").split(".")[0]
        if (package == "rlinf" or index == 0
                or path.parts[index - 1].casefold() in {"site-packages", "purelib", "platlib"}):
            return False
    return True


def inspect_zip_member(stream, size, head, name, inspect_text, inspect_file, depth=0, budget=None):
    if size > ZIP_COMPRESSED_LIMIT:
        raise ValueError(f"Compressed offline ZIP exceeds the size limit: {name}")
    if depth > 3:
        raise ValueError(f"Offline archives are nested too deeply: {name}")
    # ZIP requires seeking its central directory. Spool larger files to bounded
    # scratch storage while retaining their independent full-file source check.
    # The uncompressed member size is already known. Both streams implement
    # seekable() for zipfile on the oldest supported publisher Python (3.10).
    with (io.BytesIO() if size <= ZIP_SPOOL_THRESHOLD else tempfile.TemporaryFile()) as spool:
        read_member(stream, size, name, inspect_file, head=head, spool=spool)
        spool.seek(0)
        return inspect_nested_archive(spool, name, inspect_text, inspect_file, depth,
                                      [ZIP_EXPANDED_LIMIT] if budget is None else budget)


def inspect_nested_archive(fileobj, member_path, inspect_text, inspect_file, depth, budget):
    text_count = 0
    with zipfile.ZipFile(fileobj) as archive:
        entries = archive.infolist()
        budget[0] -= sum(entry.file_size for entry in entries)
        if budget[0] < 0:
            raise ValueError(f"Expanded offline ZIP exceeds the size limit: {member_path}")
        seen = set()
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if (not valid_zip_path(path, entry.orig_filename) or path in seen
                    or entry.file_size > ZIP_ENTRY_LIMIT):
                raise ValueError(f"Excluded nested archive path or entry size: {member_path}!/{path}")
            seen.add(path)
            if entry.is_dir():
                continue
            name = f"{member_path}!/{path}"
            with archive.open(entry) as stream:
                head = stream.read(4)
                is_zip = path.suffix.casefold() in ZIP_SUFFIXES or head in ZIP_MARKERS
                is_text = inspect_text and 0 < entry.file_size <= TEXT_LIMIT
                if is_zip:
                    text_count += inspect_zip_member(stream, entry.file_size, head, name,
                                                     inspect_text, inspect_file, depth + 1, budget)
                    continue
                data = read_member(stream, entry.file_size, name, inspect_file,
                                   capture=is_text, head=head)
            if is_text and b"\0" not in data:
                inspect_text(name, data)
                text_count += 1
    return text_count


def audit_offline_runtime(exe_path, archive_offset=0, archive_length=None,
                          inspect_text=None, inspect_file=None):
    """Check the actual embedded runtime archive without extracting it."""
    members = {}
    links = {}
    home_files = {PurePosixPath("home") / PurePosixPath(PHYSX_LIBRARY)}
    home_dirs = set(next(iter(home_files)).parents) - {PurePosixPath(".")}
    required = {PurePosixPath("runtime"), PurePosixPath("runtime/bin/python"),
                PurePosixPath("runtime/lib/python3.11/site-packages"), *home_files}
    text_count = regular_files = regular_bytes = 0
    assets = {}
    with Path(exe_path).open("rb") as stream:
        stream.seek(archive_offset)
        source = stream if archive_length is None else ArchiveRegion(stream, archive_length)
        with tarfile.open(fileobj=SingleGzipStream(source), mode="r|", bufsize=1 << 20,
                          tarinfo=StrictTarHeader) as archive:
            for info in archive:
                path = PurePosixPath(info.name)
                if (str(path) != info.name or path.is_absolute() or ".." in path.parts
                        or path in members or (path.parent != PurePosixPath(".")
                                               and members.get(path.parent) != tarfile.DIRTYPE)):
                    raise ValueError(f"Unsafe or duplicate offline archive path: {info.name}")
                if info.mode & 0o7022 or info.uid or info.gid:
                    raise ValueError(f"Unsafe offline archive permissions: {info.name}")
                if path.parts and path.parts[0] == "assets":
                    safe_asset_path(path)
                    if info.isdir():
                        members[path] = tarfile.DIRTYPE
                    elif path != PurePosixPath("assets") and info.isreg() and not info.mode & 0o111:
                        members[path] = tarfile.REGTYPE
                        regular_files += 1
                        regular_bytes += info.size
                        observe, finish = inspect_file(info.name) if inspect_file else (None, None)
                        with archive.extractfile(info) as member:
                            assets[str(path.relative_to("assets"))] = inspect_asset(
                                member, info.name, info.size, inspect_text, observe
                            )
                        if finish:
                            finish()
                        text_count += bool(inspect_text and path.suffix.casefold() in TEXT_TYPES)
                    else:
                        raise ValueError(f"Unsafe offline asset member: {path}")
                    continue
                if path in home_dirs:
                    if not info.isdir():
                        raise ValueError(f"Offline asset parent is not a directory: {path}")
                elif path in home_files:
                    if not info.isreg() or not info.size:
                        raise ValueError(f"Expected the PhysX GPU shared library: {path}")
                elif (path != PurePosixPath("runtime") and (len(path.parts) < 2
                      or path.parts[0] != "runtime" or path.parts[1] not in PREFIX_ROOTS)):
                    raise ValueError(f"Unexpected offline archive path: {path}")
                if (IGNORED.intersection(path.parts) or path.suffix in {".egg-link", ".pyc", ".pyo"}
                        or path.name.startswith("__editable__")
                        or (path.name == "direct_url.json" and path.parent.name.endswith(".dist-info"))
                        or (path.parent.name == "site-packages" and protected_package(path.name))):
                    raise ValueError(f"Excluded path in offline archive: {path}")
                if info.issym():
                    target = PurePosixPath(info.linkname)
                    if target.is_absolute() or path.parts[0] != "runtime":
                        raise ValueError(f"Unexpected offline archive symlink: {path}")
                    links[path] = target
                elif info.islnk():
                    target = PurePosixPath(info.linkname)
                    if members.get(target) != tarfile.REGTYPE or target.parts[0] != "runtime":
                        raise ValueError(f"Unexpected offline archive hard link: {path}")
                elif not (info.isdir() or info.isreg()):
                    raise ValueError(f"Unexpected offline archive member type: {path}")
                members[path] = tarfile.REGTYPE if info.isreg() else info.type
                is_import_file = path.suffix == ".pth" and path.parent.name == "site-packages"
                if is_import_file and (not info.isreg() or info.size > TEXT_LIMIT):
                    raise ValueError(f"Unsupported offline import file: {path}")
                if not info.isreg():
                    continue
                regular_files += 1
                regular_bytes += info.size
                with archive.extractfile(info) as member:
                    head = member.read(4)
                    if path in home_files and head != b"\x7fELF":
                        raise ValueError(f"Expected the PhysX GPU shared library: {path}")
                    is_zip = path.suffix.casefold() in ZIP_SUFFIXES or head in ZIP_MARKERS
                    if is_zip:
                        text_count += inspect_zip_member(member, info.size, head, info.name,
                                                         inspect_text, inspect_file)
                        continue
                    is_text = inspect_text and 0 < info.size <= TEXT_LIMIT
                    contents = read_member(member, info.size, info.name, inspect_file,
                                           capture=is_import_file or is_text, head=head)
                if is_import_file:
                    for line in contents.decode().splitlines():
                        location = line.strip()
                        if location and not location.startswith(("#", "import ", "import\t")):
                            target = posixpath.normpath(str(path.parent / location))
                            if not target.startswith("runtime/"):
                                raise ValueError(f"Nonportable offline import path: {path}")
                if is_text and b"\0" not in contents:
                    inspect_text(info.name, contents)
                    text_count += 1

            after_first_zero = archive.fileobj.tell()
            padding = 512 + (-after_first_zero - 512) % tarfile.RECORDSIZE
            if archive.fileobj.read(padding + 1) != bytes(padding):
                raise ValueError("Unexpected data or padding after the offline tar terminator")

    for link, target in links.items():
        resolved = list(link.parent.parts)
        pending = list(reversed(target.parts))
        traversals = 0
        while pending:
            part = pending.pop()
            if part == "..":
                if len(resolved) == 1:
                    raise ValueError(f"Offline archive symlink escapes runtime: {link}")
                resolved.pop()
                continue
            resolved.append(part)
            current = PurePosixPath(*resolved)
            if current in links:
                resolved.pop()
                pending.extend(reversed(links[current].parts))
                traversals += 1
                if traversals > 40:
                    raise ValueError(f"Offline archive has a cyclic symlink: {link}")
        if len(resolved) < 2 or PurePosixPath(*resolved) not in members:
            raise ValueError(f"Offline archive has a dangling symlink: {link}")
    if missing := required - members.keys():
        raise ValueError(f"Offline archive is missing: {', '.join(map(str, sorted(missing)))}")
    return {"members": len(members), "regular_files": regular_files, "regular_file_bytes": regular_bytes,
            "hashed_regular_files": regular_files if inspect_file else len(assets),
            "text_members": text_count, "text_limit_bytes": TEXT_LIMIT, "assets": assets}
