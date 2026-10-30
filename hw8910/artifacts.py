"""Verify file contents before publication and loading, not just matching metadata."""

from pathlib import Path
from release import DIGEST, digest, fingerprint


def artifact_root(manifest, workspace):
    workspace = Path(workspace).resolve()
    relative = Path(manifest.get("artifact_dir", "."))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact_dir must be a path inside workspace")
    root = (workspace / relative).resolve()
    if not root.is_relative_to(workspace):
        raise ValueError("Artifact escapes workspace")
    return root


def verify_files(manifest, workspace):
    root = artifact_root(manifest, workspace)
    hashes = manifest.get("file_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing pinned file list")
    actual = {}
    for name, expected in hashes.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not DIGEST.fullmatch(str(expected)):
            raise ValueError("Invalid artifact entry")
        file = (root / path).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError("Missing artifact: " + name)
        actual[name] = digest(file.read_bytes())
        if actual[name] != expected:
            raise ValueError("Artifact file changed: " + name)
    if fingerprint(actual) != manifest["model_sha256"]:
        raise ValueError("Artifact aggregate hash mismatch")
    if manifest.get("backend") == "transformers":
        from huggingface_hub import snapshot_download

        runtime = manifest["runtime"]
        if runtime["generation"]["max_new_tokens"] != 32 or runtime["generation"]["do_sample"] is not False:
            raise ValueError("This API supports the evaluated greedy 32-token runtime only")
        snapshot = Path(
            snapshot_download(
                repo_id=runtime["base_model"],
                revision=runtime["base_revision"],
                allow_patterns=["*.json", "*.safetensors"],
            )
        )
        for name, expected in runtime["base_files"].items():
            if Path(name).name != name or digest((snapshot / name).read_bytes()) != expected:
                raise ValueError("Base file changed: " + name)
        required = {
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "manifest.json",
        }
        if not required <= set(hashes):
            raise ValueError("Incomplete real model package")
        token_files = manifest["tokenizer_files"]
        if not token_files or not set(token_files) <= set(actual):
            raise ValueError("Invalid tokenizer file list")
        if fingerprint({k: actual[k] for k in token_files}) != manifest["runtime"]["tokenizer_sha256"]:
            raise ValueError("Tokenizer hash mismatch")
        if actual["chat_template.jinja"] != manifest["runtime"]["template_sha256"]:
            raise ValueError("Template hash mismatch")
    return {
        "artifact_uri": manifest["artifact_uri"],
        "model_sha256": fingerprint(actual),
        "files": len(actual),
        "verified": True,
    }
