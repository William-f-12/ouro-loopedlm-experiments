"""Merge complete, non-overlapping result directories or downloaded ZIPs; no GPU needed."""
import argparse
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.merging import merge_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="Batch directories or result ZIPs")
    parser.add_argument("--dataset", choices=["gsm8k", "math500"], default="math500")
    parser.add_argument("--expected-examples", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = merge_runs(args.inputs, args.output_dir, dataset=args.dataset, expected_examples=args.expected_examples)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Merge failed: {error}\n")
    print(f"Merged {args.expected_examples} unique {args.dataset} examples from {len(args.inputs)} batches.")
    print(f"Recomputed {len(report['groups'])} group summaries and {len(report['paired'])} paired comparisons.")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
