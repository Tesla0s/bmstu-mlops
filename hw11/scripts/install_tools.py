"""Download pinned local tools and verify official SHA256 files (macOS ARM64)."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
WORK=Path(os.environ.get("HW11_WORKSPACE",ROOT.parents[1]/"hw11/workspace"))
BIN=WORK/"bin"
BIN.mkdir(parents=True,exist_ok=True)
items=[
 ("kind","https://kind.sigs.k8s.io/dl/v0.30.0/kind-darwin-arm64",
  "https://github.com/kubernetes-sigs/kind/releases/download/v0.30.0/kind-darwin-arm64.sha256sum"),
 ("kubectl","https://dl.k8s.io/release/v1.34.0/bin/darwin/arm64/kubectl",
  "https://dl.k8s.io/release/v1.34.0/bin/darwin/arm64/kubectl.sha256"),
 ("helm.tar.gz","https://get.helm.sh/helm-v3.19.0-darwin-arm64.tar.gz",
  "https://get.helm.sh/helm-v3.19.0-darwin-arm64.tar.gz.sha256sum")]
verified=[]
for name,url,checksum_url in items:
    target=BIN/name
    if not target.exists():
        subprocess.run(["curl","-fL","--retry","3",url,"-o",str(target)],check=True)
    check=target.with_name(name+".sha256")
    subprocess.run(["curl","-fL","--retry","3",checksum_url,"-o",str(check)],check=True)
    expected=check.read_text().split()[0]
    actual=hashlib.sha256(target.read_bytes()).hexdigest()
    if actual!=expected:
        raise RuntimeError(f"Checksum mismatch: {name}")
    if name=="helm.tar.gz":
        with tarfile.open(target) as archive:
            member=archive.extractfile("darwin-arm64/helm")
            (BIN/"helm").write_bytes(member.read())
    else:target.chmod(0o755)
    verified.append({"file":name,"url":url,"sha256":actual,"verified":True})
(BIN/"helm").chmod(0o755)
import json
out=ROOT/"results/environment"
out.mkdir(parents=True,exist_ok=True)
(out/"tool-checksums.json").write_text(json.dumps(verified,indent=2))
print("Verified kind, kubectl, Helm against official SHA256 files.")
