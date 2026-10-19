"""Merge JSONL files in order.

Example:
  python src_report/merge_jsonl.py \
      --inputs a.jsonl b.jsonl \
      --out merged.jsonl
"""

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Merge JSONL files.")
    p.add_argument("--inputs", nargs="+", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for path in args.inputs:
            with open(path, encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line if line.endswith("\n") else line + "\n")
                        n += 1
    print(f"Wrote {args.out}")
    print(f"Records: {n}")


if __name__ == "__main__":
    main()
