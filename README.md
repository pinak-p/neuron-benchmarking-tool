# neuron-benchmarking-tool
Benchmark model performance on AWS Trainium/Inferentia instances for various configurations.

## Steps to run the tool
- **Environment Setup**
    - Create a AWS Trainium or Inferentia EC2 instance on which you want to run the benchmarks using one of the below AMIs and install the relevant libraries
        - Base Ubuntu or Amazon Linux DLAMI
            - [Setup torch-neuronx](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx) (drivers and PyTorch Neuron)
            - Install Inference Frameworks
                - [transformers-neuronx](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/transformers-neuronx/setup/index.html)
                - [neuronx distributed inference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/nxdi-setup.html#install-nxd-inference)
        - [Neuron Multi Framework DLAMI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/dlami/index.html#multi-framework-dlamis-supported)
            - [Activate the desired virtual environment](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/multiframework/multi-framework-ubuntu22-neuron-dlami.html#setup-ubuntu22-multi-framework-dlami):
                - `/opt/aws_neuronx_venv_pytorch_2_5_transformers` for `transformers_neuronx` [recommended framework for vllm]
                - `/opt/aws_neuronx_venv_pytorch_2_5_nxd_inference` for `neuronx_distributed_inference` [in Beta]
    - Install vllm
        - Use Neuron's vllm fork [recommended approach]
            ```
            git clone -b v0.6.x-neuron https://github.com/aws-neuron/upstreaming-to-vllm.git
            cd upstreaming-to-vllm
            pip install -r requirements-neuron.txt
            VLLM_TARGET_DEVICE="neuron" && pip install -e .
            ```
        - If you install vllm from the main repository, ensure that you install it from source instead of PyPI.
    - Install [llmperf](https://github.com/ray-project/llmperf)
        - You can consider using changes from this [PR](https://github.com/ray-project/llmperf/pull/81) for more accurate results by using the relevant tokenizer i.e. the one corresponding to the model.

- **Create a configuration file**
    - some sample configuration files exist in `sample_configs/`.
    - A configuration file is a `.yaml` that needs to have a list of configurations.
    - Each configuration should have the following fields
        
        ```
        - config:
            # parameters to configure vllm and compile the model
            vllm_params:
                # huggingface modelid  
                modelid:
                # sequence length for model compilation
                max_sequence_length:
                # tensor parallel degree for model sharding
                tensor_parallel_degree:
                # batch size with which the model should be compiled
                # you can set this to a number or 'sweep' (in which case the script benchmarks it for the following batch sizes: 2, 4, 8, 10, 16, 24)
                batch_size:
            # parameters to be used in the benchmarking client (llmperf)
            benchmark_params:
                # number of input tokens
                num_input_tokens: 1024
                #number of output tokens
                num_output_tokens: 512
                #number of concurrent requests in the client. When batch_size='sweep', this is set to the corresponding batch size
                concurrency: 4
                # directory to store the benchmarking results
                results_path: "test"
            # name associated with the config
            name: "config1"
        ```
- Run `python benchmark.py --config <your_config.yaml>`
    - This script stands up a vllm server using the parameters from the config file and benchmarks it using llmperf on the instance.
    - The raw llmperf results are stored at `results_path`.
    - Metrics across various configs are summarized in `neuron-vllm-results.csv` at the top-level directory.

