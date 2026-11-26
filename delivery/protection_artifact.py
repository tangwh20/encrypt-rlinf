"""Bind publisher-side protection analysis to the audited offline release."""

import hashlib
import os
from pathlib import Path
import stat

from judge_template import JUDGE_MODULE
from source_audit import (
    LAUNCHER_NAMES, LEGACY_LAUNCHER_NAME, SIDECARS, file_identity, hash_region, release_executable,
)


from source_inventory import EXTENSION_SUFFIX


def _require_names(label, expected, actual):
    expected, actual = set(expected), set(actual)
    if actual != expected:
        missing = ", ".join(sorted(expected - actual)[:5])
        extra = ", ".join(sorted(actual - expected)[:5])
        raise ValueError(f"{label} mismatch: missing [{missing}], unexpected [{extra}]")


def _file_receipt(path: Path, expected_identity=None) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular publication file: {path}")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        identity = file_identity(info)
        if not stat.S_ISREG(info.st_mode) or (expected_identity and identity != expected_identity):
            raise ValueError(f"File does not match audited publication: {path}")
        if path.name in (*LAUNCHER_NAMES, "install.sh") and not info.st_mode & 0o111:
            raise ValueError(f"Missing executable permission: {path}")
        digest = hash_region(stream, info.st_size).hex()
        if (identity != file_identity(os.fstat(stream.fileno()))
                or identity != file_identity(path.lstat())):
            raise ValueError(f"Publication file changed while hashing: {path}")
    return {"size": info.st_size, "sha256": digest}


class ArtifactInventory:
    def __init__(self):
        self._files = {}

    def observe(self, name: str, data: bytes) -> None:
        """Receive each regular member exactly once during the production audit."""
        if name in self._files:
            raise ValueError(f"Duplicate application member: {name}")
        if name.endswith(".so") and not (
            len(data) >= 64 and data[:7] == b"\x7fELF\x02\x01\x01"
            and data[16:20] == b"\x03\x00\x3e\x00"  # ET_DYN, EM_X86_64
        ):
            raise ValueError(f"Expected an x86-64 shared object: {name}")
        self._files[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    def bind(self, directory: Path, audit: dict, expected_modules: set[Path],
             expected_resources: dict[str, str]) -> dict:
        """Verify packaged business modules, helpers and resources against source inventory."""
        offline = audit.get("offline_archive", {})
        content = offline.get("content_audit", {})
        if (not content.get("regular_files")
                or content["regular_files"] != content.get("hashed_regular_files")
                or not isinstance(content.get("assets"), dict)):
            raise ValueError("Protection analysis requires a fully audited offline release")
        identity = audit.get("audited_file_identity")
        if not identity or audit.get("findings") != []:
            raise ValueError("Missing successful source audit with member observation")

        business = {
            f"app/{path.with_suffix(EXTENSION_SUFFIX).as_posix()}": path.as_posix()
            for path in expected_modules
        }
        executable = release_executable(directory)
        prefix = "_rlinf" if executable.name == LEGACY_LAUNCHER_NAME else "_train"
        helpers = {f"{name}{EXTENSION_SUFFIX}" for name in (
            f"{prefix}_release_start", "app/_rlinf_launcher", f"setup/{prefix}_check_environment"
        )}
        if JUDGE_MODULE in expected_modules:
            helpers.add(f"app/_rlinf_rstar2_sandbox{EXTENSION_SUFFIX}")
        resources = {name: sha for name, sha in expected_resources.items() if name != "NOTICE"}
        names = business.keys() | helpers | {f"app/{name}" for name in resources} | {"app/NOTICE"}
        _require_names("Application files", names, self._files)
        members = audit["members"]
        if members["types"].get("L", 0) or members["types"]["F"] != len(self._files):
            raise ValueError("Unexpected application symlinks or incomplete member observation")
        parents = {str(parent) for name in names for parent in Path(name).parents if parent != Path(".")}
        _require_names("Application directories", parents, members["directories"])
        for name, sha in resources.items():
            if self._files[f"app/{name}"]["sha256"] != sha:
                raise ValueError(f"Packaged resource differs from publisher input: {name}")

        _require_names("Offline release directory", (*SIDECARS, executable.name),
                       (path.name for path in directory.iterdir()))
        sidecars = {name: _file_receipt(directory / name) for name in SIDECARS}
        if sidecars != audit.get("release_sidecars"):
            raise ValueError("Publication sidecars differ from the audited files")
        return {
            "executable": _file_receipt(executable, identity),
            "application_payload": audit["application_payload"],
            "offline_archive": offline,
            "sidecars": sidecars,
            "module_binaries": {source: self._files[name]["sha256"]
                                for name, source in sorted(business.items())},
            "resources": {name: self._files[f"app/{name}"] for name in sorted(resources)},
            "generated_notice": self._files["app/NOTICE"],
            "publisher_helpers": {name: self._files[name] for name in sorted(helpers)},
        }
