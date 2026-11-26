"""Authenticated delivery envelopes for selected native modules. Keys travel with the offline launcher."""

from pathlib import Path
import secrets

from source_inventory import EXTENSION_SUFFIX, SOURCE_TREES

KEY_MARKER = b"RLinf-AES-key-v1"
NONCE_BYTES = 12
TAG_BYTES = 16
MAX_NATIVE = 256 << 20


def native_member(source):
    return "app/" + Path(source).with_suffix(EXTENSION_SUFFIX).as_posix()


def check_member(name):
    parts = name.split("/")
    if (len(parts) < 3 or parts[0] != "app" or parts[1] not in SOURCE_TREES
            or any(part in ("", ".", "..") for part in parts) or not name.endswith(EXTENSION_SUFFIX)):
        raise ValueError(f"Unexpected encrypted member {name!r}")


def aes(key):
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise ValueError("Install python3-cryptography on the publisher to handle encrypted RLinf modules.") from exc
    return AESGCM(key)


def launcher_key(launcher):
    pos = launcher.find(KEY_MARKER)
    if pos < 0:
        return None
    end = pos + len(KEY_MARKER)
    if end + 32 > len(launcher) or launcher.find(KEY_MARKER, end) >= 0:
        raise ValueError("Invalid native module key descriptor")
    return launcher[end:end + 32]


def prepare_key(header):
    key = secrets.token_bytes(32)
    aes(key)
    values = ", ".join(str(value) for value in KEY_MARKER + key)
    header.write_text("#define RLINF_SEALED_MODULES 1\n"
                      "static const unsigned char rlinf_key_record[] __attribute__((used)) = {"
                      + values + "};\n", encoding="ascii")
    return key


def seal_native_modules(selected, native_dir, envelopes, key):
    result = {}
    cipher = aes(key)
    for module in selected:
        name = native_member(module)
        check_member(name)
        source = native_dir / name.removeprefix("app/")
        if source.stat().st_size > MAX_NATIVE - NONCE_BYTES - TAG_BYTES:
            raise ValueError(f"Native module too large for encryption: {module}")
        data = source.read_bytes()
        if not data.startswith(b"\x7fELF"):
            raise ValueError(f"Invalid compiled module: {module}")
        nonce = secrets.token_bytes(NONCE_BYTES)
        target = envelopes / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(nonce + cipher.encrypt(nonce, data, name.encode("utf-8")))
        target.chmod(source.stat().st_mode & 0o777)
        result[name] = target
    return result


def open_member(key, name, envelope):
    check_member(name)
    if key is None or not NONCE_BYTES + TAG_BYTES < len(envelope) <= MAX_NATIVE:
        raise ValueError(f"Invalid encrypted native module: {name}")
    cipher = aes(key)
    from cryptography.exceptions import InvalidTag
    try:
        data = cipher.decrypt(envelope[:NONCE_BYTES], envelope[NONCE_BYTES:], name.encode("utf-8"))
    except InvalidTag as exc:
        raise ValueError(f"Authentication failed for encrypted module: {name}") from exc
    if not data.startswith(b"\x7fELF"):
        raise ValueError(f"Invalid authenticated native module: {name}")
    return data
