# 1. keep everything as it is
# 2. run docker compose up -d in `docker` directory
# 3. check logs with docker compose logs -f
# 4. wait for the container to start. Sample logs:
```
2025-11-24 13:54:38.392 | --- Starting setup and server ---
2025-11-24 13:54:38.392 | Modifying vllm entrypoint...
2025-11-24 13:54:38.392 | + echo '--- Starting setup and server ---'
2025-11-24 13:54:38.392 | + echo 'Modifying vllm entrypoint...'
2025-11-24 13:54:38.392 | ++ which vllm
2025-11-24 13:54:38.393 | + sed -i '/^from vllm\.entrypoints\.cli\.main import main/a from DotsOCR import modeling_dots_ocr_vllm' /usr/local/bin/vllm
2025-11-24 13:54:38.396 | vllm script after patch:
2025-11-24 13:54:38.396 | + echo 'vllm script after patch:'
2025-11-24 13:54:38.397 | ++ which vllm
2025-11-24 13:54:38.397 | + grep -A 1 'from vllm.entrypoints.cli.main import main' /usr/local/bin/vllm
2025-11-24 13:54:38.398 | from vllm.entrypoints.cli.main import main
2025-11-24 13:54:38.398 | from DotsOCR import modeling_dots_ocr_vllm
2025-11-24 13:54:38.398 | Starting server...
2025-11-24 13:54:38.398 | + echo 'Starting server...'
2025-11-24 13:54:38.398 | + exec vllm serve /workspace/weights/DotsOCR --tensor-parallel-size 1 --gpu-memory-utilization 0.8 --chat-template-content-format string --served-model-name dotsocr-model --trust-remote-code
2025-11-24 13:54:43.087 | INFO 11-23 22:54:43 [__init__.py:244] Automatically detected platform cuda.
2025-11-24 13:54:48.145 | INFO 11-23 22:54:48 [api_server.py:1287] vLLM API server version 0.9.1
2025-11-24 13:54:48.360 | INFO 11-23 22:54:48 [cli_args.py:309] non-default args: {'chat_template_content_format': 'string', 'model': '/workspace/weights/DotsOCR', 'trust_remote_code': True, 'served_model_name': ['dotsocr-model'], 'gpu_memory_utilization': 0.8}
2025-11-24 13:54:48.392 | INFO 11-23 22:54:48 [config.py:823] This model supports multiple tasks: {'embed', 'score', 'reward', 'classify', 'generate'}. Defaulting to 'generate'.
2025-11-24 13:54:48.396 | INFO 11-23 22:54:48 [config.py:2195] Chunked prefill is enabled with max_num_batched_tokens=2048.
2025-11-24 13:54:49.527 | WARNING 11-23 22:54:49 [env_override.py:17] NCCL_CUMEM_ENABLE is set to 0, skipping override. This may increase memory overhead with cudagraph+allreduce: https://github.com/NVIDIA/nccl/issues/1234
2025-11-24 13:54:50.729 | INFO 11-23 22:54:50 [__init__.py:244] Automatically detected platform cuda.
2025-11-24 13:54:52.654 | INFO 11-23 22:54:52 [core.py:455] Waiting for init message from front-end.
2025-11-24 13:54:52.660 | INFO 11-23 22:54:52 [core.py:70] Initializing a V1 LLM engine (v0.9.1) with config: model='/workspace/weights/DotsOCR', speculative_config=None, tokenizer='/workspace/weights/DotsOCR', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=131072, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=dotsocr-model, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=True, pooler_config=None, compilation_config={"level":3,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":["none"],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output"],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"max_capture_size":512,"local_cache_dir":null}
2025-11-24 13:54:53.430 | WARNING 11-23 22:54:53 [utils.py:2737] Methods determine_num_available_blocks,device_config,get_cache_block_size_bytes,initialize_cache not implemented in <vllm.v1.worker.gpu_worker.Worker object at 0x7c0e0b4bfb00>
2025-11-24 13:54:53.714 | INFO 11-23 22:54:53 [parallel_state.py:1065] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
2025-11-24 13:54:53.716 | WARNING 11-23 22:54:53 [interface.py:376] Using 'pin_memory=False' as WSL is detected. This may slow down the performance.
2025-11-24 13:54:53.981 | Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
2025-11-24 13:54:55.348 | INFO 11-23 22:54:55 [topk_topp_sampler.py:49] Using FlashInfer for top-p & top-k sampling.
2025-11-24 13:54:55.405 | INFO 11-23 22:54:55 [gpu_model_runner.py:1595] Starting to load model /workspace/weights/DotsOCR...
2025-11-24 13:54:55.533 | INFO 11-23 22:54:55 [gpu_model_runner.py:1600] Loading model from scratch...
2025-11-24 13:54:55.647 | INFO 11-23 22:54:55 [cuda.py:252] Using Flash Attention backend on V1 engine.
2025-11-24 13:54:55.749 | 
2025-11-24 13:54:55.719 | Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
2025-11-24 13:55:03.340 | 
2025-11-24 13:55:03.336 | Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:07<00:07,  7.59s/it]
2025-11-24 13:55:24.363 | 
2025-11-24 13:55:24.347 | Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:28<00:00, 15.49s/it]
2025-11-24 13:55:24.363 | 
2025-11-24 13:55:24.347 | Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:28<00:00, 14.31s/it]
2025-11-24 13:55:24.363 | 
2025-11-24 13:55:24.544 | INFO 11-23 22:55:24 [default_loader.py:272] Loading weights took 28.76 seconds
2025-11-24 13:55:24.799 | INFO 11-23 22:55:24 [gpu_model_runner.py:1624] Model loading took 5.7174 GiB and 28.981453 seconds
2025-11-24 13:55:25.023 | INFO 11-23 22:55:25 [gpu_model_runner.py:1978] Encoder cache will be initialized with a budget of 14400 tokens, and profiled with 1 image items of the maximum feature size.
2025-11-24 14:00:02.278 | INFO 11-23 23:00:02 [backends.py:462] Using cache directory: /root/.cache/vllm/torch_compile_cache/4761e4df6e/rank_0_0 for vLLM's torch.compile
2025-11-24 14:00:02.278 | INFO 11-23 23:00:02 [backends.py:472] Dynamo bytecode transform time: 3.87 s
2025-11-24 14:00:04.086 | INFO 11-23 23:00:04 [backends.py:161] Cache the graph of shape None for later use
2025-11-24 14:00:04.517 | [rank0]:W1123 23:00:04.516000 77 torch/_inductor/utils.py:1250] [0/0] Not enough SMs to use max_autotune_gemm mode
2025-11-24 14:00:16.699 | INFO 11-23 23:00:16 [backends.py:173] Compiling a graph for general shape takes 14.10 s
2025-11-24 14:00:23.369 | INFO 11-23 23:00:23 [monitor.py:34] torch.compile takes 17.97 s in total
2025-11-24 14:00:30.181 | INFO 11-23 23:00:30 [gpu_worker.py:227] Available KV cache memory: 3.65 GiB
2025-11-24 14:00:30.379 | INFO 11-23 23:00:30 [kv_cache_utils.py:715] GPU KV cache size: 136,592 tokens
2025-11-24 14:00:30.380 | INFO 11-23 23:00:30 [kv_cache_utils.py:719] Maximum concurrency for 131,072 tokens per request: 1.04x
2025-11-24 14:00:46.497 | INFO 11-23 23:00:46 [gpu_model_runner.py:2048] Graph capturing finished in 16 secs, took 0.31 GiB
2025-11-24 14:00:46.520 | INFO 11-23 23:00:46 [core.py:171] init engine (profile, create kv cache, warmup model) took 321.72 seconds
2025-11-24 14:00:46.960 | INFO 11-23 23:00:46 [loggers.py:137] Engine 000: vllm cache_config_info with initialization after num_gpu_blocks is: 8537
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [api_server.py:1349] Starting vLLM API server 0 on http://0.0.0.0:8000
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:29] Available routes are:
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /openapi.json, Methods: GET, HEAD
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /docs, Methods: GET, HEAD
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /docs/oauth2-redirect, Methods: GET, HEAD
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /redoc, Methods: GET, HEAD
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /health, Methods: GET
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /load, Methods: GET
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /ping, Methods: POST
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /ping, Methods: GET
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /tokenize, Methods: POST
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /detokenize, Methods: POST
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /v1/models, Methods: GET
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /version, Methods: GET
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /v1/chat/completions, Methods: POST
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /v1/completions, Methods: POST
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /v1/embeddings, Methods: POST
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /pooling, Methods: POST
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /classify, Methods: POST
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /score, Methods: POST
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /v1/score, Methods: POST
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /v1/audio/transcriptions, Methods: POST
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /rerank, Methods: POST
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /v1/rerank, Methods: POST
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /v2/rerank, Methods: POST
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /invocations, Methods: POST
2025-11-24 14:00:46.982 | INFO 11-23 23:00:46 [launcher.py:37] Route: /metrics, Methods: GET
2025-11-24 14:00:47.037 | INFO:     Started server process [1]
2025-11-24 14:00:47.037 | INFO:     Waiting for application startup.
2025-11-24 14:00:47.244 | INFO:     Application startup complete.
```