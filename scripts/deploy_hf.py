#!/usr/bin/env python3
"""Create and push a Hugging Face Space, end to end.

    HF_TOKEN=hf_xxx DATABASE_URL=postgresql://... python3 scripts/deploy_hf.py

Everything the platform needs is assembled here rather than by hand: the
Space, its Docker variant of the app, the secrets, and a wait on the build.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COPY = ["app", "scripts", "eval", ".streamlit", "streamlit_app.py", "requirements.txt"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rag-knowledge-assistant")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL", ""))
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    if not args.token:
        print("no token: set HF_TOKEN or pass --token")
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    who = api.whoami()["name"]
    repo_id = f"{who}/{args.name}"
    print(f"  account: {who}")

    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker",
                    private=args.private, exist_ok=True)
    print(f"  space:   https://huggingface.co/spaces/{repo_id}")

    # Secrets first: the build starts on upload, and a Space that boots without
    # its database shows an error page to anyone who happens to look.
    if args.database_url:
        api.add_space_secret(repo_id=repo_id, key="DATABASE_URL", value=args.database_url)
    for key, value in {
        "LLM_PROVIDER": "anthropic",
        "EMBEDDING_PROVIDER": "fastembed",
        "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "EMBEDDING_DIM": "384",
        "COOKIE_SECURE": "true",
        "ALLOW_SIGNUP": "true",
        "RETRIEVAL_TOP_K": "20",
        "RERANK_TOP_N": "5",
        "MIN_CONFIDENCE": "0.35",
    }.items():
        api.add_space_variable(repo_id=repo_id, key=key, value=value)
    print("  secrets and variables set")

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "space"
        stage.mkdir()
        for item in COPY:
            src = ROOT / item
            if src.is_dir():
                shutil.copytree(src, stage / item,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            elif src.exists():
                shutil.copy2(src, stage / item)
        shutil.copy2(ROOT / "deploy/hf/Dockerfile", stage / "Dockerfile")
        shutil.copy2(ROOT / "deploy/hf/README.md", stage / "README.md")
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(stage),
                          commit_message="deploy knowledge assistant")
    print("  code uploaded, build started")

    for _ in range(60):
        stage_name = api.get_space_runtime(repo_id).stage
        print(f"    {stage_name}")
        if stage_name == "RUNNING":
            print(f"\n  LIVE: https://huggingface.co/spaces/{repo_id}")
            return 0
        if stage_name in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"}:
            print(f"\n  build failed: {stage_name} -- check the Space's Logs tab")
            return 1
        time.sleep(20)
    print("  still building; check the Space page")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
