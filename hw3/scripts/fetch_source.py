#!/usr/bin/env python3
"""One-time public GitHub snapshot. Normal dvc repro is offline.

GITHUB_API_IP optionally pins the official API hostname to a DNS address for
this process only; curl still checks the certificate for api.github.com.
Credentials are passed on stdin, never in process arguments or output files.
"""

import hashlib
import ipaddress
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_params


def request(url, token, ip):
    command = [
        "curl",
        "--silent",
        "--show-error",
        "--max-time",
        "90",
        "--config",
        "-",
        "--write-out",
        "\n%{http_code}",
        url,
    ]
    if ip:
        command += ["--resolve", f"api.github.com:443:{ipaddress.ip_address(ip)}"]
    headers = 'header = "Accept: application/vnd.github+json"\n'
    if token:
        if any(x in token for x in '\r\n"'):
            raise SystemExit("Invalid credential format")
        headers += f'header = "Authorization: Bearer {token}"\n'
    for attempt in range(5):
        result = subprocess.run(command, input=headers, text=True, capture_output=True)
        if result.returncode:
            raise SystemExit(f"GitHub connection failed: {result.stderr.strip()}")
        body, code = result.stdout.rsplit("\n", 1)
        if code == "200":
            payload = json.loads(body)
            if payload.get("incomplete_results"):
                raise SystemExit("GitHub returned incomplete search results; retry this page.")
            return payload
        if code in ("403", "429", "502", "503", "504") and attempt < 4:
            print(f"HTTP {code}: retry after 60 seconds", flush=True)
            time.sleep(60)
            continue
        message = json.loads(body).get("message", code) if body.startswith("{") else code
        raise SystemExit(f"GitHub HTTP {code}: {message}")
    raise SystemExit("GitHub retry limit exceeded")


def main():
    params = load_params()
    cfg, paths = params["collect"], params["paths"]
    target = Path(paths["source"])
    if target.exists():
        raise SystemExit(
            f"{target} already exists. Keep the frozen snapshot; archive it explicitly before refreshing."
        )
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token and shutil.which("gh"):
        result = subprocess.run(
            ["gh", "auth", "token", "--hostname", "github.com"],
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            token = result.stdout.strip()
    ip = os.environ.get("GITHUB_API_IP")
    cache = Path(paths["download_cache"])
    cache.mkdir(parents=True, exist_ok=True)
    records = {}
    queries = []
    retrieved_at = datetime.now(UTC).isoformat()
    searches = [
        {
            "repository": repo,
            "category": category,
            "created_from": cfg["created_from"],
            "created_to": cfg["created_to"],
        }
        for repo, classes in cfg["repositories"].items()
        for category in classes
    ] + cfg.get("extra_searches", [])
    for search in searches:
        repo, category = search["repository"], search["category"]
        label = cfg["repositories"][repo][category]
        query = (
            f"repo:{repo} is:issue is:{cfg['fetch_state']} "
            f'label:"{label}" created:{search["created_from"]}..{search["created_to"]}'
        )
        key = hashlib.sha256(query.encode()).hexdigest()[:16]
        query_rows, total = 0, None
        limit = cfg["fetch_per_repository_class"]
        for page in range(1, (limit + cfg["page_size"] - 1) // cfg["page_size"] + 1):
            file = cache / f"{key}-{page}.json"
            if file.exists():
                payload = json.loads(file.read_text())
            else:
                url = "https://api.github.com/search/issues?" + urlencode(
                    {
                        "q": query,
                        "sort": "created",
                        "order": "desc",
                        "per_page": cfg["page_size"],
                        "page": page,
                    }
                )
                payload = request(url, token, ip)
                file.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
                time.sleep(2.2 if token else 6.2)
            total = payload["total_count"]
            items = payload["items"][: limit - query_rows]
            for issue in items:
                record = {
                    "id": f"{repo}#{issue['number']}",
                    "repository": repo,
                    "number": issue["number"],
                    "url": issue["html_url"],
                    "title": issue["title"],
                    "body": issue.get("body") or "",
                    "labels": sorted(x["name"] for x in issue["labels"]),
                    "created_at": issue["created_at"],
                    "closed_at": issue["closed_at"],
                    "updated_at": issue["updated_at"],
                    "state": issue["state"],
                    "is_pull_request": "pull_request" in issue,
                    "is_bot": issue["user"]["type"] == "Bot",
                    "retrieved_at": retrieved_at,
                }
                records[record["id"]] = record
            query_rows += len(items)
            print(f"{repo} / {category}: {query_rows} / {min(total, limit)}", flush=True)
            if len(items) < cfg["page_size"] or query_rows >= min(total, limit):
                break
        queries.append({"query": query, "total_count": total, "downloaded": query_rows})
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(records[k], ensure_ascii=False, sort_keys=True) + "\n" for k in sorted(records)
    )
    target.write_text(payload, encoding="utf-8")
    manifest = {
        "retrieved_at": retrieved_at,
        "rows": len(records),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "queries": queries,
        "selection": "latest closed issues per repository and label within fixed creation-date window; maximum 1000 per query",
        "api": "https://api.github.com/search/issues",
    }
    Path(paths["source_manifest"]).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Snapshot: {len(records)} unique issues -> {target}", flush=True)


if __name__ == "__main__":
    main()
