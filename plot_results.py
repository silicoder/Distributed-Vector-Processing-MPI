import argparse
import csv
import math
import sys
from pathlib import Path


def load_results(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"'{path}' does not exist. Run vector_sequential and vector_mpi first."
        )
    rows = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"mode", "vector_size", "processes", "seconds", "checksum"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "CSV header must contain: mode,vector_size,processes,seconds,checksum"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                mode = row["mode"].strip()
                vector_size = int(row["vector_size"])
                processes = int(row["processes"])
                seconds = float(row["seconds"])
                checksum = int(row["checksum"])
            except (TypeError, ValueError):
                print(f"Skipping malformed row {line_number} in {path}.", file=sys.stderr)
                continue
            if mode not in {"Sequential", "MPI"} or vector_size <= 0 or seconds <= 0:
                print(f"Skipping unsupported/non-positive row {line_number} in {path}.", file=sys.stderr)
                continue
            rows.append({
                "mode": mode,
                "vector_size": vector_size,
                "processes": processes,
                "seconds": seconds,
                "checksum": checksum,
            })
    if not rows:
        raise ValueError(f"No valid benchmark rows were found in '{path}'.")
    return rows


def choose_comparable_rows(rows):
    by_size = {}
    for row in rows:
        by_size.setdefault(row["vector_size"], {"Sequential": None, "MPI": None})
        # The last matching row in the CSV is the most recently appended run.
        by_size[row["vector_size"]][row["mode"]] = row

    common_sizes = [
        size for size, modes in by_size.items()
        if modes["Sequential"] is not None and modes["MPI"] is not None
    ]
    if not common_sizes:
        raise ValueError(
            "No matching vector size has both Sequential and MPI timings. "
            "Run both programs with the same vector size."
        )
    selected_size = max(common_sizes)
    selected = by_size[selected_size]
    seq_row = selected["Sequential"]
    mpi_row = selected["MPI"]
    if seq_row["checksum"] != mpi_row["checksum"]:
        raise ValueError(
            "The Sequential and MPI checksums do not match for vector size "
            f"{selected_size}: {seq_row['checksum']} vs {mpi_row['checksum']}. "
            "Check the program output before interpreting performance results."
        )
    return selected_size, seq_row, mpi_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results.csv", help="Input benchmark CSV (default: results.csv)")
    parser.add_argument("--output-dir", default="graphs", help="Directory for output graphs (default: graphs)")
    args = parser.parse_args()

    try:
        rows = load_results(Path(args.input))
        vector_size, seq_row, mpi_row = choose_comparable_rows(rows)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: Matplotlib is not installed. Install it with: python3 -m pip install matplotlib", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_seconds = seq_row["seconds"]
    mpi_seconds = mpi_row["seconds"]
    speedup = seq_seconds / mpi_seconds
    processes = mpi_row["processes"]

    # Plot 1: directly measured elapsed time.
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars = ax.bar(["Sequential (1 process)", f"MPI ({processes} processes)"], [seq_seconds, mpi_seconds])
    ax.set_title(f"Sequential vs MPI Execution Time\nVector size: {vector_size:,}")
    ax.set_ylabel("Elapsed time (seconds)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, [seq_seconds, mpi_seconds]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.6f} s",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    timing_path = output_dir / "execution_time_comparison.png"
    fig.savefig(timing_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: measured speedup (not a pre-filled illustrative value).
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bar = ax.bar([f"MPI ({processes} processes)"], [speedup])[0]
    ax.axhline(1.0, linestyle="--", linewidth=1, label="1× reference")
    ax.set_title(f"MPI Speedup over Sequential\nVector size: {vector_size:,}")
    ax.set_ylabel("Speedup (sequential time / MPI time)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{speedup:.3f}×",
            ha="center", va="bottom", fontsize=10)
    ax.legend()
    fig.tight_layout()
    speedup_path = output_dir / "speedup_comparison.png"
    fig.savefig(speedup_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Vector size: {vector_size:,}")
    print(f"Sequential: {seq_seconds:.9f} seconds")
    print(f"MPI ({processes} processes): {mpi_seconds:.9f} seconds")
    print(f"Checksum: {seq_row['checksum']} (matches)")
    print(f"Measured speedup: {speedup:.3f}x")
    print(f"Saved: {timing_path}")
    print(f"Saved: {speedup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
