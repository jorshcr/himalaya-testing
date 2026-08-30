"""Download and verify an immutable Hugging Face checkpoint artifact."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import time
from urllib.parse import quote

import httpx


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")

    expected = args.sha256.lower()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    url = (
        f"https://huggingface.co/{quote(args.repo_id, safe='/')}/resolve/main/"
        f"{quote(args.filename, safe='/')}?download=true"
    )
    headers = {"Authorization": f"Bearer {token}"}
    timeout = httpx.Timeout(600.0, connect=30.0)

    for attempt in range(1, args.retries + 1):
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                with client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()
                    with partial.open("wb") as stream:
                        for chunk in response.iter_bytes(1024 * 1024):
                            stream.write(chunk)
            actual = _sha256(partial)
            if actual != expected:
                raise RuntimeError(
                    f"SHA-256 mismatch: expected {expected}, received {actual}"
                )
            partial.replace(args.output)
            print(f"checkpoint verified: {actual}")
            return
        except (httpx.HTTPError, OSError, RuntimeError) as error:
            partial.unlink(missing_ok=True)
            if attempt == args.retries:
                raise SystemExit(
                    f"checkpoint download failed after {attempt} attempts: {error}"
                ) from error
            print(f"download attempt {attempt} failed; retrying")
            time.sleep(min(2**attempt, 30))


if __name__ == "__main__":
    main()
