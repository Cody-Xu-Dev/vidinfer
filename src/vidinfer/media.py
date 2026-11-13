"""逐帧向 ffmpeg 写入 RGB，避免额外创建整段 uint8 视频副本。"""

from pathlib import Path
import subprocess
import wave


def write_media(decoded, directory, fps=24):
    import numpy as np

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    video, audio = decoded["video"], decoded["audio"]
    if video.ndim == 5 and video.shape[0] == 1:
        video = video[0]
    if audio.ndim == 3 and audio.shape[0] == 1:
        audio = audio[0]
    if (
        video.ndim != 4
        or video.shape[-1] != 3
        or audio.ndim != 2
        or audio.shape[0] != 2
    ):
        raise ValueError("Expected video[T,H,W,3] and stereo audio[2,S]")
    frames, height, width, _ = video.shape
    wav_path, mp4_path = directory / "audio.wav", directory / "video.mp4"
    pcm = (np.clip(audio.T, -1, 1) * 32767).astype("<i2")
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(int(decoded["sampling_rate"]))
        output.writeframes(pcm.tobytes())
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-i",
        str(wav_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-threads",
        "4",
        "-shortest",
        str(mp4_path),
    ]
    with (directory / "ffmpeg.log").open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for frame in video:
                frame = (
                    np.clip(frame * 255, 0, 255).astype(np.uint8)
                    if frame.dtype.kind == "f"
                    else frame.astype(np.uint8)
                )
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"ffmpeg failed; see {directory / 'ffmpeg.log'}")
    return dict(
        video_path=str(mp4_path),
        audio_path=str(wav_path),
        video_frames=frames,
        height=height,
        width=width,
        sampling_rate=int(decoded["sampling_rate"]),
        fps=fps,
    )
