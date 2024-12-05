MODEL_TO_ARGS = {
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "enforce_eager": True,
    },
    "mistralai/Ministral-8B-Instruct-2410": {
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "enforce_eager": True,
    },
    "mistralai/Mistral-Small-Instruct-2409": {
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "enforce_eager": True,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {},
    "Qwen/Qwen2.5-32B-Instruct": {
        "max_model_len": 20000,
        "gpu_memory_utilization": 0.95,
        "enforce_eager": True,
    },
    "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4": {
        "max_model_len": 8192,
        "gpu_memory_utilization": 1.0,
        "enforce_eager": True,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 2048,
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {},
    "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4": {
        "max_model_len": 8192,
        "gpu_memory_utilization": 1.0,
        "enforce_eager": True,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 2048,
    },
    "neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic": {
        "max_model_len": 8192,
        "gpu_memory_utilization": 1.0,
        "enforce_eager": True,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 2048,
    },
}
