"""Pack your own inference files with the configured scenario contract."""

import argparse
import json
import tempfile
import zipfile
from hashlib import sha256
from pathlib import Path

from core.package import ENV_ID, PROTOCOL_VERSION, inspect_package

from .env import read_scenarios, scenario_entries


def export_policy(
    config, source, output, *, name="My policy", layout_id=None, scenario_id=None
):
    config, source, output = Path(config), Path(source), Path(output)
    settings = json.loads(config.read_text(encoding="utf-8"))
    if settings["kind"] == "team" and (layout_id or scenario_id):
        raise ValueError("A team submission must cover all configured scenarios")
    entries = scenario_entries(read_scenarios(config, layout_id, scenario_id))
    if not (source / "policy.py").is_file():
        raise ValueError("The inference folder needs policy.py with class Policy")
    files = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(
            part.startswith(".") or part == "__pycache__" for part in relative.parts
        ):
            continue
        if path.is_symlink():
            raise ValueError("Inference files must not contain symbolic links")
        if path.is_file() and path.suffix in {".py", ".safetensors", ".json", ".txt"}:
            if relative.as_posix() == "manifest.json":
                raise ValueError("manifest.json is generated from config.json")
            files[relative.as_posix()] = path.read_bytes()
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_entry": "policy.py:Policy",
        "env_id": ENV_ID,
        "observation": {"channels": 26, "axis_order": "CHW", "dtype": "float32"},
        "action_count": 6,
        "name": name,
        "scenario": entries[0]["scenario"],
        "scenario_spec_id": entries[0]["scenario_spec_id"],
        "scenarios": entries,
        "files": {key: sha256(value).hexdigest() for key, value in files.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bitlinc-interface-") as temporary:
        candidate = Path(temporary) / "submission.zip"
        with zipfile.ZipFile(candidate, "w", zipfile.ZIP_DEFLATED) as archive:
            for key, value in files.items():
                archive.writestr(key, value)
            archive.writestr(
                "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2)
            )
        inspect_package(candidate)
        with output.open("xb") as destination:
            destination.write(candidate.read_bytes())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--source", default="agent")
    parser.add_argument("--output", default="submission.zip")
    parser.add_argument("--name", default="My policy")
    parser.add_argument("--map", dest="layout_id")
    parser.add_argument("--scenario-id")
    args = parser.parse_args()
    result = export_policy(**vars(args))
    print(json.dumps({"output": args.output, "scenarios": result["scenarios"]}))


if __name__ == "__main__":
    main()
