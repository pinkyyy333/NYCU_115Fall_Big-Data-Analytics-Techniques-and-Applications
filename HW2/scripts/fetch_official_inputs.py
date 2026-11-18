"""Fetch the official anonymized cohort/test lists without republishing them in this repository."""
import argparse
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check local files without network access")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "config/official_inputs.json").read_text(encoding="utf-8"))
    expected_paths = {"scripts/cohort.txt", "submission/test_input.csv"}
    if set(manifest["files"]) != expected_paths:
        raise ValueError("Unexpected official input paths")
    base = ("https://raw.githubusercontent.com/nycubigdata-2026fall/BigData2026-Assignment2/"
            +manifest["commit"]+"/")
    for relative, expected in manifest["files"].items():
        destination = ROOT / relative
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
            print(f"Verified official input: {relative}")
            continue
        if args.verify_only:
            raise SystemExit(f"Official input missing or changed: {relative}")
        response = requests.get(base+relative, timeout=(15,60))
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != expected:
            raise ValueError(f"Official input checksum mismatch: {relative}")
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary = destination.with_suffix(destination.suffix+".part")
        temporary.write_bytes(response.content)
        temporary.replace(destination)
        print(f"Downloaded and verified official input: {relative}")


if __name__ == "__main__":
    main()
