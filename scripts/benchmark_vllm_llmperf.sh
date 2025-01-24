#!/bin/bash
model_id=${1}
cores=${2:-0-31}
max_seq_len=${3:-2048}
continuous_batch_size=${4:-32}
tp_degree=${5:-32}
benchmark_concurrency=${6}
benchmark_input=${7}
benchmark_output=${8}
results_path=${9}
result_name=${10}
neuron_framework=${11:-"transformers-neuronx"}
omp_n_threads=${12:-32}
port=${13:-8000}


export VLLM_RPC_TIMEOUT=1000000
export vLLM_MAX_LEN=${max_seq_len} 
export vLLM_MODEL_ID=${model_id}
export vLLM_CONT_BATCH_SIZE=${continuous_batch_size} 
export vLLM_TENSOR_PARALLEL_SIZE=${tp_degree}
export NEURON_RT_VISIBLE_CORES=${cores}
export MASTER_PORT=12355
export OMP_NUM_THREADS=${omp_n_threads}
#export NEURON_CONTEXT_LENGTH_BUCKETS="450"
#export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export VLLM_NEURON_FRAMEWORK="transformers-neuronx"

while true; do
    if nc -z localhost ${port} >/dev/null 2>&1; then
        ((port++))
    else
        break
    fi
done

numactl --cpunodebind=0 --membind=0 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model ${vLLM_MODEL_ID} \
        --tensor-parallel-size ${vLLM_TENSOR_PARALLEL_SIZE} \
        --max-num-seqs ${vLLM_CONT_BATCH_SIZE} \
        --max-model-len ${vLLM_MAX_LEN} \
        --block-size 8 \
        --device neuron \
        --use-v2-block-manager \
        --port ${port} &
vllm_pid=$!

while true; do
    echo "waiting for vllm server ${port}"

    # Try to connect to the server
    if nc -z localhost ${port} >/dev/null 2>&1; then
        echo "vLLM Server is ready!"
        break
    fi

    # Wait before next attempt
    sleep 5
done

echo "Proceeding to benchmark ..." 

export LLM_PERF_CONCURRENCY=${benchmark_concurrency}
export LLM_PERF_MAX_REQUESTS=$(expr ${LLM_PERF_CONCURRENCY} \* 20 )
export OPENAI_API_KEY=secret_abcdefg
export OPENAI_API_BASE="http://localhost:${port}/v1"
export LLM_PERF_SCRIPT_DIR=$HOME/benchmark/llmperf
export INPUT_TOKENS=${benchmark_input}
export OUTPUT_TOKENS=${benchmark_output}
export date_str=$(date '+%Y-%m-%d-%H-%M-%S')
export LLM_PERF_OUTPUT=${results_path}/${vLLM_CONT_BATCH_SIZE}_${date_str}
 

python3 ${LLM_PERF_SCRIPT_DIR}/token_benchmark_ray.py \
    --model ${vLLM_MODEL_ID} \
    --mean-input-tokens ${INPUT_TOKENS} \
    --stddev-input-tokens 0 \
    --mean-output-tokens ${OUTPUT_TOKENS} \
    --stddev-output-tokens 0 \
    --max-num-completed-requests ${LLM_PERF_MAX_REQUESTS} \
    --timeout 36000 \
    --num-concurrent-requests ${LLM_PERF_CONCURRENCY} \
    --results-dir "${LLM_PERF_OUTPUT}" \
    --llm-api openai \
    --additional-sampling-params '{"ignore_eos": "True"}'\
    --metadata "name=${result_name}"

kill ${vllm_pid}