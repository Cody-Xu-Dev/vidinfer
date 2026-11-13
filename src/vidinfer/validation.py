"""轻量输入检查，在加载大型组件和提交 GPU 工作之前执行。"""


def validate_generation_request(request, config):
    if not isinstance(request.prompt, str) or not request.prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if request.output_type not in ("latent", "video"):
        raise ValueError("output_type must be latent or video")
    if len(request.prompt) > config.max_prompt_chars:
        raise ValueError("Prompt exceeds configured character limit")
    if any(
        type(v) is not int
        for v in (
            request.height,
            request.width,
            request.frames,
            request.seed,
            request.steps,
        )
    ):
        raise ValueError("Dimensions, seed and steps must be integers")
    if (
        request.height < 32
        or request.width < 32
        or request.height % 32
        or request.width % 32
    ):
        raise ValueError("H3 canvas dimensions must be positive multiples of 32")
    ratio = request.width / request.height
    if not 0.25 <= ratio <= 4:
        raise ValueError("H3 aspect ratio must be between 1:4 and 4:1")
    if request.height * request.width > config.max_canvas_pixels:
        raise ValueError("Canvas pixel budget exceeded")
    if request.frames < 5 or request.frames % 17 != 5:
        raise ValueError("H3 frame count must be 17*n+5 (e.g. 124,141,158)")
    if not 120 <= request.frames <= 360:
        raise ValueError(
            "Official H3 duration must be 5..15 seconds at 24fps; aligned frames are 124..345"
        )
    latent_frames = (request.frames - 5) // 17 * 5 + 2
    tokens = latent_frames * (request.height // 32) * (request.width // 32)
    if tokens > config.max_video_tokens:
        raise ValueError(
            f"Video token budget exceeded: {tokens}>{config.max_video_tokens}"
        )
    # 小画布但超长时长也会产生大量音频 Token，不能只限制视频 Token。
    audio_tokens = 2 * round(request.frames / 24 * 40)
    if tokens + audio_tokens + 2048 > config.max_total_tokens:
        raise ValueError("Total video/audio/text token budget exceeded")
    if request.steps not in (5, 29):
        raise ValueError("Prepared AdaLN schedules support 5 or 29 points only")
    if not 0 <= request.seed < 2**63:
        raise ValueError("seed must be in [0,2**63)")
    config.output_directory(request.request_id)
