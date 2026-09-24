"""Benchmark the full pipeline through the running API.

    python3 tests/benchmark.py file1.mp3 [file2.wav ...]

Env: API_URL (default http://localhost:8080), API_KEY (required).
Prints MEASURED numbers only - nothing is assumed.
"""
import os
import sys
import time
import uuid
from pathlib import Path

import requests


def main() -> None:
    files = sys.argv[1:]
    if not files:
        print("Usage: python3 tests/benchmark.py <audio> [<audio> ...]")
        sys.exit(1)
    url = os.getenv("API_URL", "http://localhost:8080").rstrip("/")
    key = os.environ.get("API_KEY")
    if not key:
        print("Set API_KEY first, e.g.  export API_KEY=$(grep ^API_KEY .env | cut -d= -f2)")
        sys.exit(1)

    rows = []
    for path in files:
        call_id = f"bench-{Path(path).stem[:40]}-{uuid.uuid4().hex[:6]}".replace(" ", "_")
        t0 = time.time()
        with open(path, "rb") as f:
            r = requests.post(f"{url}/v1/process", headers={"X-API-Key": key},
                              data={"call_id": call_id}, files={"file": f}, timeout=3600)
        wall = time.time() - t0
        body = r.json()
        if r.status_code != 200 or not body.get("success"):
            print(f"{path}: FAILED ({r.status_code}) {body.get('error') or body}")
            continue
        t = body["timings"]
        rows.append(t)
        print("-" * 50)
        print(f"File            : {path}")
        print(f"Audio duration  : {t['audio_duration_seconds']} seconds")
        print(f"STT processing  : {t['stt_seconds']} seconds")
        print(f"LLM processing  : {t['llm_seconds']} seconds")
        print(f"Total (worker)  : {t['total_seconds']} seconds")
        print(f"Total (HTTP)    : {wall:.1f} seconds")
        print(f"RTF             : {t['real_time_factor']}")

    if rows:
        audio = sum(r["audio_duration_seconds"] or 0 for r in rows)
        total = sum(r["total_seconds"] for r in rows)
        print("=" * 50)
        print(f"Files: {len(rows)}  audio: {audio:.0f}s  processing: {total:.0f}s  "
              f"overall RTF: {total / audio:.3f}" if audio else "")


if __name__ == "__main__":
    main()
