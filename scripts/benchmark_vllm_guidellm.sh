#!/bin/bash
model_id=${1}
cores=${2:-0-31}
max_seq_len=${3:-2048}
batch_size=${4:-32}
tp_degree=${5:-32}
benchmark_rps=${6}
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
export vLLM_CONT_BATCH_SIZE=${batch_size} 
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

export INPUT_TOKENS=${benchmark_input}
export OUTPUT_TOKENS=${benchmark_output}
export date_str=$(date '+%Y-%m-%d-%H-%M-%S')
 
guidellm \
  --target "http://localhost:${port}/v1" \
  --model ${vLLM_MODEL_ID} \
  --data-type emulated \
  --data "prompt_tokens=${INPUT_TOKENS},generated_tokens=${OUTPUT_TOKENS},prompt_tokens_variance=0,generated_tokens_variance=0"\
  --rate-type constant \
  --rate ${benchmark_rps} \
  --max-requests 1000 \
  --output-path ${results_path}/${vLLM_CONT_BATCH_SIZE}_${benchmark_rps}_${INPUT_TOKENS}_${OUTPUT_TOKENS}_${date_str}.json

kill ${vllm_pid}