benchmark_dataset_b.py
from benchmark import run_benchmark


if __name__ == "__main__":
    run_benchmark(
        chunks_path="data/processed/chunks_dataset_b.json",
        output_path="results/benchmark_results_dataset_b.json",
        dataset="b"
    )