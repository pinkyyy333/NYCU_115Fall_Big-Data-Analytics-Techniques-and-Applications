"""Download original GH Archive hours; retain source bytes, checksums and missing-hour records."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time

import requests


def hours(start, end):
    current = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    stop = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    if current.minute or current.second or stop.minute or stop.second:
        raise ValueError("Use whole UTC hours")
    while current < stop:
        yield current
        current += timedelta(hours=1)


def download(start, end, out, workers=4, max_gb=60):
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder.parent / (folder.name + "_manifest.jsonl")
    lock = threading.Lock()
    used = sum(p.stat().st_size for p in folder.glob("*.json.gz"))
    initial = time.monotonic()
    complete = 0

    def fetch(hour):
        nonlocal used, complete
        name = f"{hour:%Y-%m-%d}-{hour.hour}.json.gz"
        target = folder / name
        url = "https://data.gharchive.org/" + name
        result = {"file": name, "url": url, "status": "failed"}
        if target.exists():
            result.update(status="cached", bytes=target.stat().st_size)
        else:
            for attempt in range(3):
                try:
                    with lock:
                        if used >= max_gb * 2**30:
                            raise RuntimeError("Configured download budget reached")
                    if shutil.disk_usage(folder).free < 10 * 2**30:
                        raise RuntimeError("Less than 10 GiB free disk space")
                    with requests.get(url, stream=True, timeout=(15, 90)) as response:
                        result["http_status"] = response.status_code
                        if response.status_code == 404:
                            result["status"] = "missing"
                            break
                        response.raise_for_status()
                        digest, size = hashlib.sha256(), 0
                        partial = target.with_suffix(target.suffix + ".part")
                        with partial.open("wb") as handle:
                            for chunk in response.iter_content(1024 * 1024):
                                if not chunk:
                                    continue
                                if size == 0 and not chunk.startswith(b"\x1f\x8b"):
                                    raise ValueError("Response is not gzip data")
                                handle.write(chunk)
                                digest.update(chunk)
                                size += len(chunk)
                        expected = response.headers.get("Content-Length")
                        if expected and size != int(expected):
                            raise ValueError(f"Truncated download: {size}/{expected}")
                        partial.replace(target)
                        with lock:
                            used += size
                        result.update(status="downloaded", bytes=size, sha256=digest.hexdigest())
                        break
                except Exception as exc:
                    result["error"] = str(exc)
                    if attempt < 2:
                        time.sleep(2 ** attempt)
        result["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        with lock:
            complete += 1
            with manifest.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")
            elapsed = time.monotonic() - initial
            print(f"[{complete}] {name}: {result['status']} "
                  f"{result.get('bytes', 0)/2**20:.1f} MiB; "
                  f"total {used/2**30:.2f} GiB, {elapsed/60:.1f} min", flush=True)
        return result

    requested = list(hours(start, end))
    if not requested:
        raise ValueError("End must be later than start")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(fetch, requested))
    counts = {s: sum(r["status"] == s for r in results)
              for s in ["downloaded", "cached", "missing", "failed"]}
    print(json.dumps(counts), flush=True)
    if counts["failed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="Inclusive UTC timestamp")
    parser.add_argument("--end", required=True, help="Exclusive UTC timestamp")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-gb", type=float, default=60)
    args = parser.parse_args()
    download(args.start, args.end, args.out, args.workers, args.max_gb)


if __name__ == "__main__":
    main()
