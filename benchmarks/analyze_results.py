"""Parse benchmark JSON and print a Markdown summary table."""
import argparse
import json
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, default=pathlib.Path("benchmarks/results/backend_comparison.json"))
    args = parser.parse_args()

    if not args.input.exists():
        print(f"No results file found: {args.input}")
        print("Run: make benchmark")
        return

    rows = json.loads(args.input.read_text())

    headers = ["Backend", "Batch", "Concurrency", "p50 ms", "p95 ms", "p99 ms", "Tput req/s", "Errors"]
    col_widths = [max(len(h), 18) for h in headers]

    def row_str(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, col_widths)) + " |"

    print("\n## Benchmark Results\n")
    print(row_str(headers))
    print("| " + " | ".join("-" * w for w in col_widths) + " |")

    for r in rows:
        cells = [
            r.get("backend", ""),
            r.get("batch_size", ""),
            r.get("concurrency", ""),
            f"{r.get('p50_ms', 0):.1f}",
            f"{r.get('p95_ms', 0):.1f}",
            f"{r.get('p99_ms', 0):.1f}",
            f"{r.get('throughput_rps', 0):.1f}",
            r.get("errors", 0),
        ]
        print(row_str(cells))

    print()
    md_path = args.input.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("## Benchmark Results\n\n")
        f.write(row_str(headers) + "\n")
        f.write("| " + " | ".join("-" * w for w in col_widths) + " |\n")
        for r in rows:
            cells = [
                r.get("backend", ""),
                r.get("batch_size", ""),
                r.get("concurrency", ""),
                f"{r.get('p50_ms', 0):.1f}",
                f"{r.get('p95_ms', 0):.1f}",
                f"{r.get('p99_ms', 0):.1f}",
                f"{r.get('throughput_rps', 0):.1f}",
                r.get("errors", 0),
            ]
            f.write(row_str(cells) + "\n")
    print(f"Markdown table written to {md_path}")


if __name__ == "__main__":
    main()
