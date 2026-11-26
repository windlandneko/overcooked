"""Portable, standard-library-only validation of student policy archives."""

import hashlib
import json
import re
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

PROTOCOL_VERSION = "bitlinc-policy-v1"
ENV_ID = "overcooked-739950a-single-order-v1"
# The current course may accept less; runtime validation must keep older,
# already accepted packages usable after an administrator lowers that limit.
DEFAULT_MODEL_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_ZIP_BYTES = 10 * 1024 * 1024
MAX_EXPANDED_BYTES = 20 * 1024 * 1024
MAX_FILES = 100


class PackageError(ValueError):
    pass


def manifest_scenarios(manifest):
    """One policy entry can declare every scene it supports; legacy ZIPs declare one."""
    first = {
        "scenario": manifest.get("scenario"),
        "scenario_spec_id": manifest.get("scenario_spec_id"),
    }
    scenarios = manifest.get("scenarios", [first])
    if not isinstance(scenarios, list) or not 1 <= len(scenarios) <= 12:
        raise PackageError("模型包需要声明 1–12 个支持的场景")
    seen = set()
    for item in scenarios:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("scenario"), dict)
            or not isinstance(item.get("scenario_spec_id"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["scenario_spec_id"])
        ):
            raise PackageError("缺少有效的场景配置和指纹")
        if item["scenario_spec_id"] in seen:
            raise PackageError("模型包场景指纹不能重复")
        seen.add(item["scenario_spec_id"])
    if any(scenarios[0].get(key) != value for key, value in first.items()):
        raise PackageError("默认场景必须与支持场景列表的第一项一致")
    return scenarios


def supports_scenario(manifest, scenario, fingerprint):
    return any(
        item["scenario_spec_id"] == fingerprint and item["scenario"] == scenario
        for item in manifest_scenarios(manifest)
    )


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PackageError("manifest 存在重复字段")
        result[key] = value
    return result


def _entries(archive, *, ignore_desktop_files=False):
    infos = archive.infolist()
    if not 1 <= len(infos) <= MAX_FILES:
        raise PackageError("模型包最多包含 100 个文件和目录")
    seen, entries, total = set(), [], 0
    for info in infos:
        name = info.filename
        path = PurePosixPath(name)
        if ignore_desktop_files and (
            path.name == ".DS_Store" or path.parts[:1] == ("__MACOSX",)
        ):
            continue
        if (
            not name
            or "\\" in name
            or "\x00" in name
            or path.is_absolute()
            or any(part in {"..", "."} or part.startswith(".") for part in path.parts)
            or str(path) != name.rstrip("/")
            or ":" in name
        ):
            raise PackageError("模型包包含无效路径")
        canonical = name.rstrip("/").casefold()
        if canonical in seen:
            raise PackageError("模型包存在重复文件名")
        seen.add(canonical)
        mode = info.external_attr >> 16
        if (
            stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}
            or info.flag_bits & 1
        ):
            raise PackageError("模型包不能包含链接、特殊文件或加密文件")
        if info.is_dir():
            continue
        if path.suffix.lower() not in {
            ".py",
            ".safetensors",
            ".json",
            ".txt",
            ".md",
        } and path.name not in {"LICENSE", "NOTICE"}:
            raise PackageError("仅支持 Python、safetensors 和文本配置文件")
        total += info.file_size
        if total > MAX_EXPANDED_BYTES:
            raise PackageError(f"模型包展开后不能超过 {MAX_EXPANDED_BYTES / (1024 * 1024):g} MiB")
        entries.append(info)
    return entries


def prepare_submission(zip_path, scenarios, name="My policy"):
    """Add server-owned metadata to a plain student ZIP, without importing code.

    Existing exported packages keep their declarations and must still pass the
    strict validator. A single enclosing folder from Finder/Explorer is accepted.
    """
    path = Path(zip_path)
    if path.stat().st_size > MAX_ZIP_BYTES:
        raise PackageError(f"模型包不能超过 {MAX_ZIP_BYTES / (1024 * 1024):g} MiB")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = _entries(archive, ignore_desktop_files=True)
            names = {entry.filename for entry in entries}
            prefix = ""
            if "policy.py" not in names:
                roots = {item.split("/", 1)[0] for item in names}
                if len(roots) == 1:
                    candidate = next(iter(roots)) + "/"
                    if candidate + "policy.py" in names:
                        prefix = candidate
                if not prefix:
                    raise PackageError("ZIP 内需要 policy.py；也可以将包含它的文件夹直接压缩")
            # Leave an already exported archive byte-for-byte unchanged.
            if not prefix and "manifest.json" in names and len(entries) == len(archive.infolist()):
                return inspect_package(path)
            files, total = {}, 0
            for entry in entries:
                content = archive.read(entry)
                total += len(content)
                if total > MAX_EXPANDED_BYTES:
                    raise PackageError(f"模型包展开后不能超过 {MAX_EXPANDED_BYTES / (1024 * 1024):g} MiB")
                files[entry.filename[len(prefix):]] = content
    except (zipfile.BadZipFile, EOFError, RuntimeError, UnicodeError) as exc:
        raise PackageError("请上传完整有效的 ZIP 文件") from exc
    if "manifest.json" not in files:
        manifest = {
            "protocol_version": PROTOCOL_VERSION,
            "policy_entry": "policy.py:Policy",
            "env_id": ENV_ID,
            "observation": {"channels": 26, "axis_order": "CHW", "dtype": "float32"},
            "action_count": 6,
            "name": name,
            **scenarios[0],
            "scenarios": scenarios,
            "files": {key: hashlib.sha256(value).hexdigest() for key, value in files.items()},
        }
        files["manifest.json"] = json.dumps(manifest, ensure_ascii=False).encode()
    # Validate before replacing the incoming archive. Downstream workers receive
    # the same immutable protocol regardless of the student's packaging method.
    with tempfile.TemporaryDirectory(prefix="bitlinc-package-", dir=path.parent) as temporary:
        candidate = Path(temporary) / "submission.zip"
        with zipfile.ZipFile(candidate, "w", zipfile.ZIP_DEFLATED) as archive:
            for filename, content in files.items():
                archive.writestr(filename, content)
        manifest = inspect_package(candidate)
        candidate.chmod(0o600)
        candidate.replace(path)
    return manifest


def inspect_package(zip_path):
    path = Path(zip_path)
    if path.stat().st_size > MAX_ZIP_BYTES:
        raise PackageError(f"模型包不能超过 {MAX_ZIP_BYTES / (1024 * 1024):g} MiB")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = _entries(archive)
            by_name = {info.filename: info for info in entries}
            if "manifest.json" not in by_name or "policy.py" not in by_name:
                raise PackageError("模型包根目录需要 manifest.json 和 policy.py")
            if by_name["manifest.json"].file_size > 64 * 1024:
                raise PackageError("manifest 文件过大")
            manifest = json.loads(
                archive.read("manifest.json"), object_pairs_hook=_object
            )
            if not isinstance(manifest, dict):
                raise PackageError("manifest 必须是 JSON 对象")
            if (
                manifest.get("protocol_version") != PROTOCOL_VERSION
                or manifest.get("env_id") != ENV_ID
            ):
                raise PackageError("模型协议或环境版本不匹配，请使用当前教学示例导出")
            if manifest.get("policy_entry") != "policy.py:Policy":
                raise PackageError("推理入口必须是 policy.py:Policy")
            if (
                manifest.get("observation")
                != {"channels": 26, "axis_order": "CHW", "dtype": "float32"}
                or type(manifest.get("action_count")) is not int
                or manifest["action_count"] != 6
            ):
                raise PackageError("需要 26 通道 CHW float32 观测和 6 个离散动作")
            manifest_scenarios(manifest)
            files = manifest.get("files")
            if not isinstance(files, dict) or set(files) != set(by_name) - {
                "manifest.json"
            }:
                raise PackageError("manifest 文件清单与实际文件不一致")
            actual_total = 0
            for name, expected in files.items():
                if not isinstance(expected, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", expected
                ):
                    raise PackageError("文件 SHA256 格式不正确")
                digest = hashlib.sha256()
                with archive.open(name) as source:
                    while chunk := source.read(1024 * 1024):
                        actual_total += len(chunk)
                        if actual_total > MAX_EXPANDED_BYTES:
                            raise PackageError(f"模型包展开后不能超过 {MAX_EXPANDED_BYTES / (1024 * 1024):g} MiB")
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise PackageError("模型文件校验失败，请重新导出")
            return manifest
    except (
        zipfile.BadZipFile,
        UnicodeError,
        json.JSONDecodeError,
        EOFError,
        RuntimeError,
    ) as exc:
        raise PackageError("模型包不是完整有效的 ZIP/JSON 文件") from exc


def extract_package(zip_path, destination):
    manifest = inspect_package(zip_path)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()) or target.is_symlink():
        raise PackageError("解压目标必须是独立空目录")
    with zipfile.ZipFile(zip_path) as archive:
        for info in _entries(archive):
            path = target / info.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, path.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            path.chmod(0o400)
    return manifest
