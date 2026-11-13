# Third-party dependencies and model assets

- MiniMax-H3 model architecture, modular pipeline and schedulers are imported from Diffusers. No vendor model source is copied into this package. Diffusers upstream source carries its own license: https://github.com/huggingface/diffusers
- Qwen3-VL text encoding uses Transformers and Accelerate. Their licenses apply independently: https://github.com/huggingface/transformers and https://github.com/huggingface/accelerate
- Tensor operations and extension builds use PyTorch: https://github.com/pytorch/pytorch
- Checkpoint I/O uses Safetensors: https://github.com/huggingface/safetensors
- The CUDA implementations in `src/vidinfer/kernels` are the custom kernels developed for this project. NVIDIA CUDA/cuDNN and PyTorch runtime requirements apply separately.
- Media export invokes the user's ffmpeg installation; no ffmpeg binary is distributed here: https://ffmpeg.org
- Model weights, tokenizer/processor assets, precomputed tensors and generated media are not included. Users obtain the official checkpoint separately and follow the applicable MiniMax-H3 model license: https://huggingface.co/MiniMaxAI/MiniMax-H3

Configuration and local checkpoint access are explicit. No authentication tokens, model download credentials or machine-specific model paths are distributed.
