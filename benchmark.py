import os
import argparse
import yaml
import subprocess
import time
import glob
import json
import pandas as pd
from guidellm.core import GuidanceReport

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


def rps_benchmark(benchmark_configs: dict[str, any]) -> None:
    args = [
        '/bin/bash',
        "scripts/benchmark_vllm_guidellm.sh",
        benchmark_configs['vllm_params']['modelid'],
        "0-31",
        str(benchmark_configs['vllm_params']['max_sequence_length']),
        str(benchmark_configs['vllm_params']['batch_size']),
        str(benchmark_configs['vllm_params']['tensor_parallel_degree']),
        str(benchmark_configs['benchmark_params']['rps']),
        str(benchmark_configs['benchmark_params']['num_input_tokens']),
        str(benchmark_configs['benchmark_params']['num_output_tokens']),
        benchmark_configs['benchmark_params']['results_path'],
        benchmark_configs['name']
    ]
    try:
        process = subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print("Error message:", e.stderr)

def generate_csv_for_concurrency_benchmarks(
    concurrency_benchmark_results_path: str
) -> None:

    #get unique paths if the same path was used for multiple configs
    concurrency_benchmark_results_paths = set(concurrency_benchmark_results_paths)
    
    results = []

    for path in concurrency_benchmark_results_paths:
        result_dirs = glob.glob(f"{path}/*/")
        for dir in result_dirs:
            result_file = glob.glob(f"{dir}/*_summary.json")[0]
            with open(result_file, 'r') as f:
                result = json.load(f)
            results.append(pd.DataFrame([result]))
    final_df = pd.concat(results)
    final_df.to_csv("neuron-vllm-llmperf-results.csv", index=False)

def generate_csv_for_rps_benchmarks(
    rps_benchmark_results_paths: str
) -> None:

    #get unique paths if the same path was used for multiple configs
    rps_benchmark_results_paths = set(rps_benchmark_results_paths)
    results = []

    for path in rps_benchmark_results_paths:
        files = glob.glob(f"{path}/*.json")
        for file in files:
            with open(file) as f:
                model_id = file.split('/')[-1]
                report = GuidanceReport.from_json(f.read()) #json.load(f)
                for benchmark in report.benchmarks:
                    for b in benchmark.benchmarks_sorted:
                        d = {
                                "model_id": benchmark.args['model'],
                                "Input Requests per Second": b.rate,
                                "Completed Requests per Second": b.completed_request_rate,
                                "Request Latency (s)": b.request_latency,
                                "Time-to-first-token (ms)": b.time_to_first_token,
                                "Inter Token Latency (ms)": b.inter_token_latency,
                                "Output Token Throughput (t/s)": b.output_token_throughput,
                            }
                        results.append(pd.DataFrame.from_dict(d, orient="index").transpose())
    final_df = pd.concat(results)
    final_df.to_csv("neuron-vllm-guidellm-results.csv", index=False)


def benchmark(config_file: str) -> None:
    benchmark_configs = load_config(config_file)
    
    concurrency_results = []
    rps_results = []

    for config in benchmark_configs:
        batch_size_config = config['config']['vllm_params']['batch_size']
        
        assert(batch_size_config=="sweep" or isinstance(batch_size_config, int)), "Set batch size to either an integer or 'sweep' (for sweeping through multiple batch sizes)"
        
        batch_sizes = [2, 4, 8, 10, 16, 24] if batch_size_config=="sweep" else [batch_size_config]
        
        for b in batch_sizes:
            config['config']['vllm_params']['batch_size'] = b
            if 'concurrency' in config['config']['benchmark_params'].keys():
                config['config']['benchmark_params']['concurrency'] = b
                concurrency_benchmark(config['config'])
                time.sleep(60)
                concurrency_results.append(config['config']['benchmark_params']['results_path'])
            elif 'rps' in config['config']['benchmark_params'].keys():
                rps_benchmark(config['config'])
                time.sleep(60)
                rps_results.append(config['config']['benchmark_params']['results_path'])

    if concurrency_results:
        generate_csv_for_concurrency_benchmarks(concurrency_results)
    
    if rps_results:
        generate_csv_for_rps_benchmarks(rps_results)

    
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
