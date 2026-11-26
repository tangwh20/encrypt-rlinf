"""Publisher-only comparison of complete offline release inputs with a public Git tree."""

from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

from launcher import public_scripts
from offline_assets import asset_entries
from protection_diff import build_public_symbol_index, compare_module
from protection_inputs import prepare_application_sources
from public_baseline import DEFAULT_PUBLIC_COMMIT, DEFAULT_PUBLIC_URL, read_public_snapshot, select_public_merge_base
from source_audit import LAUNCHER_NAMES, file_identity, hash_region
from source_inventory import SOURCE_TREES, inventory


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def asset_receipt(path):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"Expected a regular offline asset: {path}")
        digest = hash_region(stream, info.st_size).hex()
        if file_identity(info) != file_identity(path.lstat()):
            raise ValueError(f"Offline asset changed while hashing: {path}")
    return {"size": info.st_size, "sha256": digest}


def file_status(current, public):
    if public is None:
        return "added"
    return "identical" if current == public else "modified"


def source_commit(source):
    if not (source / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "--verify", "HEAD"], capture_output=True,
        env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
    )
    # A source export is also a valid release input. The per-file hashes describe
    # the actual working files and are authoritative even in a Git checkout.
    return result.stdout.decode("ascii").strip() if result.returncode == 0 else None


def public_file_key(path, digest):
    suffix = Path(path).suffix
    return (".yaml" if suffix == ".yml" else suffix, digest)


def public_indexes(snapshots):
    files = defaultdict(dict)
    definitions = defaultdict(list)
    for commit, snapshot in snapshots.items():
        for name, data in sorted(snapshot.items()):
            files[public_file_key(name, sha256(data))].setdefault(commit, []).append(name)
        for fingerprint, matches in build_public_symbol_index(snapshot).items():
            definitions[fingerprint].extend(dict(match, commit=commit) for match in matches)
    return files, definitions


def file_matches(index, path, digest):
    matches = index.get(public_file_key(path, digest))
    if not matches:
        return {}
    paths = sorted({name for names in matches.values() for name in names})
    return {"public_content_paths": paths[:3], "public_content_path_count": len(paths),
            "public_content_matches": [{"commit": commit, "paths": names[:3]}
                                       for commit, names in matches.items()]}


def analyze_inputs(source, public_repo, revision, task, assets=None):
    try:
        selected = revision
        public_tip = None
        if revision == "auto":
            selected, public_tip = select_public_merge_base(source, public_repo)
        commit, public = read_public_snapshot(public_repo, selected)
        snapshots = {commit: public}
        if public_tip and public_tip != commit:
            tip_commit, tip_sources = read_public_snapshot(public_repo, public_tip)
            snapshots[tip_commit] = tip_sources
    except ValueError as exc:
        raise ValueError(f"{exc}. Supply a local public checkout or bare mirror with "
                         "--public-repo DIR and select its public commit with --public-ref REV.") from exc
    if not all(any(name.startswith(tree + "/") and name.endswith(".py") for name in public)
               for tree in SOURCE_TREES):
        raise ValueError(f"Public reference {commit} does not contain the RLinf application trees.")
    public_hashes = {name: sha256(data) for name, data in public.items()}
    public_modules = {name for name in public if name.endswith(".py")}
    public_files, known_symbols = public_indexes(snapshots)
    modules, resources = inventory(source)
    rows = {}
    with tempfile.TemporaryDirectory(prefix="rlinf-protection-inputs-") as tmp:
        staged = Path(tmp) / "staged"
        manifest_path = Path(tmp) / "snapshot.json"
        prepare_application_sources(source, staged, modules, source_manifest=manifest_path)
        snapshot = json.loads(manifest_path.read_text(encoding="utf-8"))
        compiled_hashes = snapshot["staged"]
        for path in sorted(modules):
            name = path.as_posix()
            current = (source / path).read_bytes()
            digest = sha256(current)
            if digest != snapshot["original"][name]:
                raise ValueError(f"Source changed while comparing the working tree: {name}. Rerun analysis.")
            baseline = public.get(name)
            row = compare_module(name, current, baseline, public_symbols=known_symbols)
            row.update(source_sha256=digest, public_sha256=public_hashes.get(name),
                       prepared_sha256=compiled_hashes[name],
                       adapted_by_publisher=compiled_hashes[name] != digest,
                       prepared_matches_public=compiled_hashes[name] == public_hashes.get(name))
            if row["status"] != "identical":
                matches = file_matches(public_files, name, digest)
                if matches:
                    reason = "完整文件在公开快照中已存在，仅复核接入方式。"
                    row.update(matches, priority="low", reasons=[reason])
                    for symbol in row["symbols"]:
                        if symbol["priority"] in ("high", "review"):
                            symbol.update(priority="low", reasons=[reason])
            rows[name] = row

    resource_rows = {}
    for path in sorted(resources):
        name = path.as_posix()
        data = (source / path).read_bytes()
        resource_rows[name] = {"status": file_status(data, public.get(name)),
                               "source_sha256": sha256(data), "public_sha256": public_hashes.get(name)}
        if resource_rows[name]["status"] != "identical":
            resource_rows[name].update(file_matches(public_files, name, resource_rows[name]["source_sha256"]))

    registered = set(public_scripts())
    shell_paths = {p.relative_to(source).as_posix()
                   for tree in SOURCE_TREES for p in (source / tree).rglob("*.sh")
                   if not any(part.startswith(".") for part in p.relative_to(source).parts)}
    shell_rows = {}
    for name in sorted(shell_paths | {n for n in public if n.endswith(".sh")}):
        if name in shell_paths:
            data = (source / name).read_bytes()
            state = file_status(data, public.get(name))
        else:
            state = "deleted"
        if state != "identical":
            shell_rows[name] = {"status": state, "registered_launcher": name in registered}
            if name in shell_paths:
                shell_rows[name].update(file_matches(public_files, name, sha256(data)))
    module_status = Counter(row["status"] for row in rows.values())
    priorities = Counter(row["priority"] for row in rows.values())
    deleted_resources = sorted(set(public) - public_modules - resource_rows.keys()
                               - {name for name in public if name.endswith(".sh")})
    task_hash = sha256(task.read_bytes())
    task_matches = file_matches(public_files, "example.yaml", task_hash)
    plan = {
        "schema_version": 1, "mode": "source_plan", "profile": "complete_offline",
        "baseline": {"commit": commit, "requested_ref": revision, "public_tip": public_tip,
                     "selection": "common_ancestor" if public_tip else "explicit_or_pinned_commit",
                     "official_public_url": DEFAULT_PUBLIC_URL if commit == DEFAULT_PUBLIC_COMMIT else None,
                     "identity": "pinned_public_commit" if commit == DEFAULT_PUBLIC_COMMIT else "publisher_selected_local_ref"},
        "source": {"checkout": source.name, "head": source_commit(source),
                   "comparison_input": "actual_working_tree_files"},
        "summary": {"business_modules": len(rows), "source_status": dict(module_status),
                    "priorities": dict(priorities), "public_files_identical_after_preparation":
                        sum(row["prepared_matches_public"] for row in rows.values()),
                    "publisher_adapted_modules": sum(row["adapted_by_publisher"] for row in rows.values()),
                    "readable_upstream_files": len(resource_rows)},
        "prepared_hash_basis": {"mode": "publisher_replay", "python": sys.version.split()[0]},
        "modules": rows, "removed_public_modules": sorted(public_modules - rows.keys()),
        "readable_resources": resource_rows, "removed_public_resources": deleted_resources,
        "upstream_shell_changes_not_shipped_as_source": shell_rows,
        "example_source": {"name": task.name, "source_sha256": task_hash,
                           "matches_public_resource": bool(task_matches), **task_matches},
        "planned_data_assets": {name.relative_to("assets").as_posix(): asset_receipt(path)
                                for name, path in asset_entries(assets) if path.is_file()} if assets else {},
    }
    plan["automatic_hardening"] = {"status": "planned", "selection": "priority == high",
                                  "modules": sorted(name for name, row in rows.items() if row["priority"] == "high")}
    return plan


def attach_release(plan, artifact, manifest=None):
    if manifest is not None:
        rows = plan["modules"]
        original = {name: row["source_sha256"] for name, row in rows.items()}
        if manifest.get("original") != original or set(manifest.get("staged", {})) != rows.keys():
            raise ValueError("Protection analysis checkout differs from the compiler's private input snapshot.")
        for name, digest in manifest["staged"].items():
            row = rows[name]
            row.update(prepared_sha256=digest, adapted_by_publisher=digest != row["source_sha256"],
                       prepared_matches_public=digest == row["public_sha256"])
        plan["summary"]["publisher_adapted_modules"] = sum(row["adapted_by_publisher"] for row in rows.values())
        plan["summary"]["public_files_identical_after_preparation"] = sum(row["prepared_matches_public"] for row in rows.values())
        plan["prepared_hash_basis"] = {"mode": "actual_compiler_snapshot", "python": "3.11"}
    plan["mode"] = "verified_build" if manifest is not None else "existing_release"
    plan["source_binding"] = ("compilation_input_hashes_verified" if manifest is not None
                              else "replayed_from_supplied_checkout_without_build_manifest")
    plan["artifact"] = artifact


def check_destination(directory, release, protected=()):
    if directory.is_symlink() or directory.exists():
        raise ValueError(f"Protection report directory already exists: {directory}. Choose a new directory.")
    _check_report_location(directory.resolve(), release, protected)


def check_audit_destination(path, release, protected=()):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"Expected a regular source audit report file: {path}")
    _check_report_location(path.resolve(), release, protected)


def _check_report_location(location, release, protected):
    if location.is_relative_to(release) or release.is_relative_to(location):
        raise ValueError("Keep publisher reports separate from the delivery directory.")
    # Keep publisher evidence out of installed releases and the default runtime, including
    # independently named older releases and prepared prefixes anywhere up the path.
    for parent in (location, *location.parents):
        if parent.name == ".git":
            raise ValueError("Keep publisher reports outside Git metadata.")
        if (parent / "install.sh").is_file() and any((parent / name).is_file() for name in LAUNCHER_NAMES):
            raise ValueError(f"Keep publisher reports outside the existing release: {parent}")
        if (parent / "lib/python3.11/site-packages").is_dir() and (parent / "bin/python").is_file():
            raise ValueError(f"Keep publisher reports outside the runtime prefix: {parent}")
    for root in protected:
        if root is not None and location.is_relative_to(root.resolve()):
            raise ValueError(f"Choose a publisher report destination outside {root}.")


def markdown(report):
    modules = report["modules"]
    counts = report["summary"]
    kinds = counts["source_status"]
    modes = {"verified_build": "已核对实际编译输入与最终离线制品",
             "existing_release": "已核对现有离线制品，源码归属根据提供的工作树重放推断",
             "source_plan": "构建前预览，尚未核对最终离线制品"}
    lines = ["# 离线交付的差异与保护优先级", "",
             f"公开基线 `{report['baseline']['commit']}`。{modes[report['mode']]}。",
             "", f"完整交付包含 {len(modules)} 个业务模块。按同路径与公开基线比较，"
             f"{kinds.get('identical', 0)} 个原文相同，{kinds.get('added', 0)} 个新增，"
             f"{kinds.get('modified', 0)} 个有逻辑变化或需人工核验，{kinds.get('text_only', 0)} 个只有文本变化。"
             f"构建兼容层另改写了 {counts['publisher_adapted_modules']} 个模块。",
             "", "优先级由可观察的代码特征推断。业务价值由项目方确认，实际复刻难度需要独立盲测。"
             "客户能访问离线程序的输入、输出和本地运行环境。", ""]
    hardening = report.get("automatic_hardening", {})
    if hardening:
        count = len(hardening["modules"])
        message = {
            "planned": f"本次计划对 {count} 个高优先级模块使用强化原生编译和 AES-256-GCM 封装。",
            "verified": f"已将 {count} 个模块的强化编译产物与最终 AES-256-GCM 归档逐项核对。",
            "observed_archive_only": f"在该成品中实际核验到 {count} 个 AES-256-GCM 封装模块。原生编译选项需由构建回执确认。",
        }[hardening["status"]]
        lines += [message, ""]
    if report["mode"] == "existing_release":
        lines += ["旧制品的模块目录和公开资源可核对。仅凭编译后的扩展无法证明它来自当前私有 Python 源文。", ""]
    tip = report["baseline"]["public_tip"]
    if tip and tip != report["baseline"]["commit"]:
        lines += [f"文件增删改以共同祖先为准。公开文件和定义的匹配还包括镜像提交 `{tip}`。", ""]
    if report.get("artifact"):
        artifact = report["artifact"]
        runtime = artifact["offline_archive"]["content_audit"]
        lines += [f"发行 ELF SHA-256 为 `{artifact['executable']['sha256']}`。", ""]
        if "hashed_regular_files" in runtime:
            lines += [f"已核对 {len(artifact['module_binaries'])} 个业务扩展、"
                      f"{len(artifact['publisher_helpers'])} 个发布辅助扩展，"
                      f"并校验离线环境中 {runtime['hashed_regular_files']:,} 个常规文件。"
                      "这些环境文件的完整性检查不判断第三方依赖是否经过私有改写。", ""]
    for priority, title, guidance in (
            ("high", "优先保护", "优先核对这些新增或改变的决策及状态更新。可用保留行为样例和公开基线盲测复核其实际保护效果。"),
            ("review", "人工复核", "核对新增或变动的行为是否承载自研规则。"),
            ("low", "较低优先级", "检查文档暴露、接口变化或可直接用公开实现解释的部分。")):
        group = [(name, row) for name, row in modules.items() if row["priority"] == priority]
        lines += [f"## {title}（{len(group)}）", "", guidance, ""]
        for name, row in group:
            reasons = " ".join(row["reasons"])
            lines.append(f"- `{name}`　{reasons}")
            for symbol in row["symbols"]:
                if symbol.get("priority") == "high":
                    lines.append(f"  - `{symbol['qualified_name']}`（第 {symbol['current_line']} 行）")
        lines.append("")
    lines += ["## 明文配置、数据及入口", "",
              "YAML、运行数据、许可证、客户安装脚本和随包提供的模型或数据均可被接收方读取。"
              "请复查变动中是否写入专有规则、模型或数据；用户需要调整的参数仍可继续公开。", ""]
    for name, row in report["readable_resources"].items():
        if row["status"] != "identical":
            lines.append(f"- `{name}`　{row['status']}")
    if "delivered_example" in report:
        example = report["delivered_example"]
        paths = ", ".join(example["matching_checkout_yaml"]) or "外部或自定义 YAML，请复核公开内容"
        lines += ["", f"客户示例 `example.yaml` 对应 {paths}。"]
    else:
        lines += ["", f"计划交付的客户示例来自 `{report['example_source']['name']}`。"]
    assets = (report["artifact"]["offline_archive"]["content_audit"]["assets"]
              if "artifact" in report else report["planned_data_assets"])
    lines += ["", f"附带的模型和数据共 {len(assets)} 个文件。"]
    if assets:
        lines += ["这些资产以客户可提取的形式离线交付，应单独确认使用范围。", ""]
        lines += [f"- `{name}`（{row['size']} 字节）" for name, row in sorted(assets.items())]
    lines += ["", "上游 Shell 实现不随默认离线包原样分发。下面的差异可能需要同步到编译启动器。", ""]
    for name, row in report["upstream_shell_changes_not_shipped_as_source"].items():
        extra = "，已注册入口" if row["registered_launcher"] else ""
        lines.append(f"- `{name}`　{row['status']}{extra}")
    lines += ["", "完整的文件与符号清单、删除项、构建适配、指纹及成品附带资产见 [report.json](report.json)。", ""]
    return "\n".join(lines)


def write_report_files(directory, report):
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "priorities.md").write_text(markdown(report), encoding="utf-8")


@contextmanager
def report_draft(directory, report):
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".protection-report-") as tmp:
        draft = Path(tmp) / "report"
        draft.mkdir(mode=0o700)
        write_report_files(draft, report)
        if directory.exists() or directory.is_symlink():
            raise FileExistsError(directory)
        yield draft
