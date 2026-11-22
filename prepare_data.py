"""Download the ETT benchmark CSVs (ETTh1, ETTm1) into data/ and record sha256.

The ETT dataset (https://github.com/zhouhaoyi/ETDataset) is CC BY-ND 4.0.
This repo does NOT redistribute it: the files are fetched at run time.
"""
import hashlib
import json
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
FILES = ["ETTh1.csv", "ETTm1.csv"]
DATA = Path(__file__).parent / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    summary = {}
    for name in FILES:
        path = DATA / name
        if not path.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(BASE + name, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        summary[name] = {"sha256": digest, "bytes": path.stat().st_size, "source": BASE + name}
        print(name, digest)
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
