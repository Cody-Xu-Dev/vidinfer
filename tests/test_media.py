"""真实 ffmpeg 小视频测试；核心 CI 没装媒体依赖时明确跳过。"""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("numpy")
    and shutil.which("ffmpeg")
    and shutil.which("ffprobe"),
    "Requires NumPy, ffmpeg and ffprobe",
)
class MediaTests(unittest.TestCase):
    def test_export_contains_rgb_video_and_stereo_audio(self):
        import numpy as np
        from vidinfer.media import write_media

        with tempfile.TemporaryDirectory() as folder:
            decoded = dict(
                video=np.full((1, 6, 32, 32, 3), 0.5, dtype=np.float32),
                audio=np.zeros((2, 8000), dtype=np.float32),
                sampling_rate=32000,
            )
            result = write_media(decoded, Path(folder) / "output")
            run = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-of",
                    "json",
                    result["video_path"],
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            streams = json.loads(run.stdout)["streams"]
            video = next(v for v in streams if v["codec_type"] == "video")
            audio = next(v for v in streams if v["codec_type"] == "audio")
            self.assertEqual(
                (video["width"], video["height"], int(video["nb_frames"])), (32, 32, 6)
            )
            self.assertEqual(audio["channels"], 2)
            self.assertEqual(video["pix_fmt"], "yuv420p")
            with self.assertRaises(FileExistsError):
                write_media(decoded, Path(folder) / "output")

    def test_invalid_media_shape_is_rejected(self):
        import numpy as np
        from vidinfer.media import write_media

        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            write_media(
                dict(
                    video=np.zeros((32, 32)),
                    audio=np.zeros((2, 10)),
                    sampling_rate=32000,
                ),
                Path(folder) / "invalid",
            )
