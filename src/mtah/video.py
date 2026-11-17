"""Local-only, frame-exact access to the tactical video using its presentation timestamps."""

from fractions import Fraction

import av

from mtah.paths import FPS, N_FRAMES


class FrameReader:
    def __init__(self, path, expected_frames=N_FRAMES):
        self.container = av.open(str(path))
        try:
            self.stream = self.container.streams.video[0]
            self.n_frames = self.stream.frames
            if self.n_frames != expected_frames or self.stream.average_rate != FPS:
                raise ValueError("video frame count or FPS differs from the selected reference")
            if self.stream.start_time != 0:
                raise ValueError("video must start at presentation timestamp zero")
            self.stream.thread_type = "AUTO"
            self.decoder = self.container.decode(self.stream)
            self.next_index = 0
        except Exception:
            self.container.close()
            raise

    def read(self, index):
        if type(index) is not int or not 0 <= index < self.n_frames:
            raise ValueError(f"frame must be an integer in 0..{self.n_frames - 1}")
        if index != self.next_index:
            target_pts = int(Fraction(index, FPS) / self.stream.time_base)
            self.container.seek(target_pts, stream=self.stream, backward=True, any_frame=False)
            self.decoder = self.container.decode(self.stream)
        for frame in self.decoder:
            if frame.pts is None:
                raise ValueError("decoded frame has no presentation timestamp")
            actual = Fraction(frame.pts) * frame.time_base * FPS
            if actual.denominator != 1:
                raise ValueError("decoded timestamp is not on the reference frame grid")
            if actual < index:
                continue
            if actual > index:
                raise ValueError(f"video skipped requested frame {index}; no frame was substituted")
            self.next_index = index + 1
            return frame
        raise ValueError(f"video ended before frame {index}")

    def close(self):
        self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
