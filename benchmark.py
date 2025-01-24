import os
import argparse
import yaml
import subprocess
import time
import glob
import json
import pandas as pd

def concurrency_benchmark(benchmark_configs: dict[str, any]) -> None:
    
    args = [
        '/bin/bash',
        "scripts/benchmark_vllm_llmperf.sh",
        benchmark_configs['vllm_params']['modelid'],
        "0-31",
        str(benchmark_configs['vllm_params']['max_sequence_length']),
        str(benchmark_configs['vllm_params']['batch_size']),
        str(benchmark_configs['vllm_params']['tensor_parallel_degree']),
        str(benchmark_configs['benchmark_params']['concurrency']),
        str(benchmark_configs['benchmark_params']['num_input_tokens']),
        str(benchmark_configs['benchmark_params']['num_output_tokens']),
        benchmark_configs['benchmark_params']['results_path'],
        benchmark_configs['name']
    ]
    try:
        process = subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print("Error message:", e.stderr)


def rps_benchmark():
    pass

def generate_csv(
    concurrency_benchmark_results_paths: str
) -> None:

    #get unique paths if the same path was used for multiple configs
    concurrency_benchmark_results_paths = set(concurrency_benchmark_results_paths)
    
    results = []

    for path in concurrency_benchmark_results_paths:
        result_dirs = glob.glob(f"{path}/*/")
        for dir in result_dirs:
            result_file = glob.glob(f"{dir}/*_summary.json")[0]
            print(result_file)
            with open(result_file, 'r') as f:
                result = json.load(f)
            results.append(pd.DataFrame([result]))
    final_df = pd.concat(results)
    final_df.to_csv("neuron-vllm-results.csv", index=False)


def benchmark(config_file: str) -> None:
    benchmark_configs = load_config(config_file)
    
    concurrency_results = []

    for config in benchmark_configs:
        batch_size_config = config['config']['vllm_params']['batch_size']
        assert(batch_size_config=="sweep" or isinstance(batch_size_config, int)), "Set batch size to either an integer or 'sweep' (for sweeping through multiple batch sizes)"
        batch_sizes = [2, 4, 8, 10, 16, 24] if batch_size_config=="sweep" else [batch_size_config]
        for b in batch_sizes:
            config['config']['vllm_params']['batch_size'] = b
            config['config']['benchmark_params']['concurrency'] = b
            concurrency_benchmark(config['config'])
        time.sleep(60)
        concurrency_results.append(config['config']['benchmark_params']['results_path'])
    generate_csv(concurrency_results)

    print("Results: --> ", concurrency_results)
    
def load_config(file_path: str) -> dict[str, any]:
    try:
        with open(file_path) as config_file:
            configs = yaml.safe_load(config_file)
    except Exception as e:
        raise e
    return configs

def main():
    args = argparse.ArgumentParser(
        description="Run benchmarks measuring token throughput and latencies."
    )

    args.add_argument(
        "--config", type=str, required=True, help="The yaml file with configurations for benchmarking."
    )
    args = args.parse_args()
    benchmark(args.config)


if __name__ == "__main__":
    main()
