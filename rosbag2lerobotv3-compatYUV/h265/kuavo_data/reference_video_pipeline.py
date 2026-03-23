from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from typing import Any

from h265_rewriter import H265StreamRewriter, START_CODE


NAL_UNIT_CODED_SLICE_BLA_W_LP = 16
NAL_UNIT_CODED_SLICE_CRA = 21
NAL_UNIT_CODED_SLICE_IDR_W_RADL = 19
NAL_UNIT_CODED_SLICE_IDR_N_LP = 20
NAL_UNIT_VPS = 32
NAL_UNIT_SPS = 33
NAL_UNIT_PPS = 34


def iter_annexb_nalus(data: bytes):
    i = 0
    size = len(data)
    while i < size - 4:
        start_code = 0
        if data[i:i + 4] == b"\x00\x00\x00\x01":
            start_code = 4
        elif data[i:i + 3] == b"\x00\x00\x01":
            start_code = 3
        if start_code:
            nal_start = i + start_code
            nal_end = size
            j = nal_start
            while j < size - 4:
                if data[j:j + 4] == b"\x00\x00\x00\x01" or data[j:j + 3] == b"\x00\x00\x01":
                    nal_end = j
                    break
                j += 1
            yield data[nal_start:nal_end]
            i = nal_end
        else:
            i += 1


def _nal_unit_type(nalu: bytes) -> int | None:
    if len(nalu) < 2:
        return None
    return (nalu[0] >> 1) & 0x3F


def is_keyframe(data: bytes) -> bool:
    for nalu in iter_annexb_nalus(data):
        nal_type = _nal_unit_type(nalu)
        if nal_type is None:
            continue
        if NAL_UNIT_CODED_SLICE_BLA_W_LP <= nal_type <= NAL_UNIT_CODED_SLICE_CRA:
            return True
        if nal_type in (NAL_UNIT_VPS, NAL_UNIT_SPS, NAL_UNIT_PPS):
            return True
    return False


def is_idr_frame(data: bytes) -> bool:
    for nalu in iter_annexb_nalus(data):
        nal_type = _nal_unit_type(nalu)
        if nal_type in (NAL_UNIT_CODED_SLICE_IDR_W_RADL, NAL_UNIT_CODED_SLICE_IDR_N_LP):
            return True
    return False


def extract_h265_headers(data: bytes) -> bytes:
    headers = bytearray()
    for nalu in iter_annexb_nalus(data):
        nal_type = _nal_unit_type(nalu)
        if nal_type in (NAL_UNIT_VPS, NAL_UNIT_SPS, NAL_UNIT_PPS):
            headers.extend(START_CODE)
            headers.extend(nalu)
        if nal_type is not None and NAL_UNIT_CODED_SLICE_BLA_W_LP <= nal_type <= NAL_UNIT_CODED_SLICE_CRA:
            break
    return bytes(headers)


class ReferenceH265VideoEncoder:
    def __init__(self, filename: str | Path, fps: int):
        self.filename = str(filename)
        self.fps = int(fps)
        self.stream_path = Path(f"{self.filename}.h265")
        self.mp4_path = Path(f"{self.filename}.mp4")
        self.writer = H265StreamRewriter(self.stream_path, Path(self.filename).name)
        self.extradata: bytes = b''
        self.opened = False

    def set_extradata(self, extradata: bytes) -> None:
        self.extradata = extradata or b''

    def open(self) -> None:
        if self.opened:
            return
        self.stream_path.parent.mkdir(parents=True, exist_ok=True)
        if self.extradata:
            self.writer.write_payload(self.extradata)
        self.opened = True

    def write_frame(self, data: bytes, is_duplicate: bool = False) -> bool:
        if not self.opened:
            self.open()
        return self.writer.write_payload(data)

    def close(self) -> Path | None:
        self.writer.close()
        if not self.stream_path.exists() or self.stream_path.stat().st_size == 0:
            return None
        commands = [
            [
                'ffmpeg', '-nostdin', '-y', '-loglevel', 'error',
                '-probesize', '100M', '-analyzeduration', '100M',
                '-r', str(self.fps), '-i', str(self.stream_path),
                '-c', 'copy', '-f', 'mp4', '-tag:v', 'hvc1', str(self.mp4_path),
            ],
            [
                'ffmpeg', '-nostdin', '-y', '-loglevel', 'error',
                '-probesize', '100M', '-analyzeduration', '100M',
                '-r', str(self.fps), '-i', str(self.stream_path),
                '-c:v', 'libx265', '-preset', 'ultrafast', '-crf', '23',
                '-f', 'mp4', '-tag:v', 'hvc1', str(self.mp4_path),
            ],
            [
                'ffmpeg', '-nostdin', '-y', '-loglevel', 'error',
                '-probesize', '100M', '-analyzeduration', '100M',
                '-r', str(self.fps), '-i', str(self.stream_path),
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
                '-f', 'mp4', str(self.mp4_path),
            ],
        ]
        for cmd in commands:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if result.returncode == 0 and self.mp4_path.exists() and self.mp4_path.stat().st_size > 0:
                try:
                    self.stream_path.unlink()
                except OSError:
                    pass
                return self.mp4_path
        return None


@dataclass
class CameraFrame:
    timestamp: float
    payload: bytes
    is_h265: bool = True
    is_duplicate: bool = False


@dataclass
class CameraSyncState:
    head_camera: str
    observation_topics: dict[str, str]
    max_time_diff: float
    current_state: dict[str, CameraFrame] = field(default_factory=dict)
    camera_first_idr_seen: dict[str, bool] = field(default_factory=dict)
    last_valid_frame: dict[str, CameraFrame] = field(default_factory=dict)
    output_started: bool = False

    def update_camera(self, camera_key: str, timestamp: float, payload: bytes, is_h265: bool = True) -> None:
        frame = CameraFrame(timestamp=timestamp, payload=payload, is_h265=is_h265, is_duplicate=False)
        self.current_state[camera_key] = frame
        if is_h265 and is_idr_frame(payload):
            self.camera_first_idr_seen[camera_key] = True

    def all_cameras_ready(self) -> bool:
        for camera_key in self.observation_topics:
            if not self.camera_first_idr_seen.get(camera_key, False):
                return False
        return True

    def try_build_aligned_frame(self, timestamp: float) -> dict[str, CameraFrame] | None:
        head_frame = self.current_state.get(self.head_camera)
        if head_frame is None:
            return None
        if not self.all_cameras_ready():
            return None
        if not self.output_started:
            if head_frame.is_h265 and not is_idr_frame(head_frame.payload):
                return None
            self.output_started = True

        aligned: dict[str, CameraFrame] = {self.head_camera: CameraFrame(timestamp, head_frame.payload, head_frame.is_h265, False)}
        for camera_key in self.observation_topics:
            if camera_key == self.head_camera:
                continue
            current = self.current_state.get(camera_key)
            if current is not None and abs(current.timestamp - timestamp) <= self.max_time_diff:
                aligned[camera_key] = CameraFrame(current.timestamp, current.payload, current.is_h265, False)
                if (not current.is_h265) or (not is_keyframe(current.payload)) or camera_key not in self.last_valid_frame:
                    self.last_valid_frame[camera_key] = current
                continue
            cached = self.last_valid_frame.get(camera_key)
            if cached is None:
                return None
            aligned[camera_key] = CameraFrame(cached.timestamp, cached.payload, cached.is_h265, True)
        return aligned
