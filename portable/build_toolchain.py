"""Copy Ubuntu's GNU toolchain package files into a relocatable sysroot."""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


PACKAGES = (
    "gcc-11", "gcc-11-base", "gcc-12-base", "g++-11", "cpp-11",
    "binutils-common", "binutils-x86-64-linux-gnu", "libcc1-0",
    "libbinutils", "libctf0", "libctf-nobfd0", "libc6", "libc6-dev",
    "linux-libc-dev", "libgcc-11-dev", "libstdc++-11-dev", "libgcc-s1",
    "libstdc++6", "libgomp1", "libatomic1", "libitm1", "libasan6", "liblsan0",
    "libtsan0", "libubsan1", "libquadmath0", "libisl23", "libmpc3", "libmpfr6",
    "libgmp10", "zlib1g", "libzstd1",
)
TOOLS = {
    "gcc": "gcc-11",
    "g++": "g++-11",
    "cc": "gcc-11",
    "c++": "g++-11",
    "cpp": "cpp-11",
    "ld": "ld.bfd",
    "as": "as",
    "ar": "ar",
    "nm": "nm",
    "ranlib": "ranlib",
    "strip": "strip",
}
WRAPPER = """#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$(realpath -- "$0")")/.." && pwd -P)
export LD_LIBRARY_PATH="$root/host-libs"
unset GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH
exec "$root/sysroot/usr/bin/{program}" {flags}"$@"
"""


def copy_package_file(root: Path, path: Path) -> None:
    if not path.is_symlink() and not path.is_file():
        return
    if not str(path).startswith((
        "/usr/bin/", "/usr/include/", "/usr/lib/", "/usr/share/doc/",
        "/usr/share/common-licenses/",
        "/lib/", "/lib64/",
    )):
        return
    output = root / path.relative_to("/")
    output.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if output.is_symlink():
            output.unlink()
        target = path.readlink()
        if target.is_absolute():
            target = Path(os.path.relpath(root / target.relative_to("/"), output.parent))
        output.symlink_to(target)
    elif not output.exists():
        shutil.copy2(path, output)


def build(directory: Path) -> None:
    sysroot = directory / "sysroot"
    package_manifest = subprocess.check_output(
        ["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\n",
         *PACKAGES],
        text=True,
    )
    for package in PACKAGES:
        paths = subprocess.check_output(["dpkg-query", "-L", package], text=True)
        for path in paths.splitlines():
            copy_package_file(sysroot, Path(path))
    for license_file in Path("/usr/share/common-licenses").iterdir():
        copy_package_file(sysroot, license_file)
    compiler_libraries = directory / "host-libs"
    compiler_libraries.mkdir()
    library_patterns = (
        "libbfd-*.so", "libopcodes-*.so", "libctf.so.0", "libctf-nobfd.so.0",
        "libisl.so.23", "libmpc.so.3", "libmpfr.so.6", "libgmp.so.10",
        "libz.so.1", "libzstd.so.1",
    )
    for pattern in library_patterns:
        matches = list((sysroot / "usr/lib/x86_64-linux-gnu").glob(pattern))
        matches.extend((sysroot / "lib/x86_64-linux-gnu").glob(pattern))
        if not matches:
            raise FileNotFoundError(pattern)
        library = matches[0]
        (compiler_libraries / library.name).symlink_to(
            os.path.relpath(library, compiler_libraries)
        )
    (directory / "packages.tsv").write_text(
        "package\tversion\tsource_package\tsource_version\n" + package_manifest
    )

    target = "x86_64-linux-gnu"
    public_bin = directory / "bin"
    public_bin.mkdir()
    private_bin = sysroot / "usr/bin"
    for name, implementation in TOOLS.items():
        program = f"{target}-{implementation}"
        if not (private_bin / program).is_file():
            raise FileNotFoundError(private_bin / program)
        flags = ""
        if name in {"gcc", "g++", "cc", "c++", "cpp"}:
            flags = '--sysroot="$root/sysroot" -B"$root/sysroot/usr/bin/" '
        elif name == "ld":
            flags = '--sysroot="$root/sysroot" '
        wrapper = public_bin / name
        wrapper.write_text(WRAPPER.format(program=program, flags=flags))
        wrapper.chmod(0o755)
        alias = private_bin / name
        if not alias.is_symlink() and not alias.exists():
            alias.symlink_to(program)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="new toolchain directory")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix=".toolchain-build-", dir=destination.parent) as tmp:
        staging = Path(tmp) / "toolchain"
        staging.mkdir()
        build(staging)
        staging.rename(destination)


if __name__ == "__main__":
    main()
