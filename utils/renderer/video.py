import moviepy.editor as mp

# import moviepy.video.fx.all as vfx
import os


class Video:
    def __init__(self, frame_path: str, fps: float = 20.0, res="high"):
        frame_path = str(frame_path)
        self.fps = fps

        self._conf = {
            "codec": "libx264",
            "fps": self.fps,
            "audio_codec": "aac",
            "temp_audiofile": "temp-audio.m4a",
            "remove_temp": True,
        }

        if res == "low":
            bitrate = "500k"
        else:
            bitrate = "5000k"

        self._conf = {"bitrate": bitrate, "fps": self.fps}

        # Load video
        # video = mp.VideoFileClip(video1_path, audio=False)
        # Load with frames
        frames = [os.path.join(frame_path, x) for x in sorted(os.listdir(frame_path))]
        video = mp.ImageSequenceClip(frames, fps=fps)
        self.video = video
        self.duration = video.duration

    def save(self, out_path):
        out_path = str(out_path)
        self.video.subclip(0, self.duration).write_videofile(
            out_path, verbose=False, logger=None, **self._conf
        )
