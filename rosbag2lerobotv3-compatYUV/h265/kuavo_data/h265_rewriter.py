from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


START_CODE = b"\x00\x00\x00\x01"


class BitstreamReader:
    def __init__(self, data: bytes):
        self.data = data
        self.size = len(data)
        self.bit_offset = 0

    def read_bits(self, n: int) -> int:
        val = 0
        for _ in range(n):
            byte_idx = self.bit_offset // 8
            bit_idx = 7 - (self.bit_offset % 8)
            if byte_idx >= self.size:
                return val
            val = (val << 1) | ((self.data[byte_idx] >> bit_idx) & 1)
            self.bit_offset += 1
        return val

    def read_ue(self) -> int:
        leading_zeros = 0
        while self.bit_offset // 8 < self.size:
            byte_idx = self.bit_offset // 8
            bit_idx = 7 - (self.bit_offset % 8)
            if (self.data[byte_idx] >> bit_idx) & 1:
                break
            leading_zeros += 1
            self.bit_offset += 1
            if leading_zeros > 31:
                return 0
        self.bit_offset += 1
        if leading_zeros == 0:
            return 0
        val = self.read_bits(leading_zeros)
        return (1 << leading_zeros) - 1 + val

    def skip_bits(self, n: int) -> None:
        self.bit_offset += n


class H265PocRewriter:
    def __init__(self) -> None:
        self.log2_max_poc_lsb_minus4 = 0
        self.separate_colour_plane_flag = False
        self.chroma_format_idc = 1
        self.log2_min_coding_block_size_minus3 = 0
        self.log2_diff_max_min_coding_block_size = 0
        self.pic_width_in_luma_samples = 0
        self.pic_height_in_luma_samples = 0
        self.sps_parsed = False

        self.dependent_slice_segments_enabled_flag = False
        self.output_flag_present_flag = False
        self.num_extra_slice_header_bits = 0
        self.pps_parsed = False

        self.pic_size_in_ctbs_y = 0
        self.slice_segment_address_bits = 0

    @staticmethod
    def remove_emulation_prevention(data: bytes) -> bytes:
        rbsp = bytearray()
        i = 0
        size = len(data)
        while i < size:
            if i + 2 < size and data[i] == 0 and data[i + 1] == 0 and data[i + 2] == 0x03:
                rbsp.extend((0, 0))
                i += 3
                continue
            rbsp.append(data[i])
            i += 1
        return bytes(rbsp)

    @staticmethod
    def add_emulation_prevention(data: bytes) -> bytes:
        out = bytearray()
        zero_count = 0
        for b in data:
            if zero_count >= 2 and b <= 0x03:
                out.append(0x03)
                zero_count = 0
            out.append(b)
            if b == 0:
                zero_count += 1
            else:
                zero_count = 0
        return bytes(out)

    @staticmethod
    def is_slice_nalu(nal_unit_type: int) -> bool:
        return 0 <= nal_unit_type <= 21

    @staticmethod
    def is_idr(nal_unit_type: int) -> bool:
        return nal_unit_type in (19, 20)

    @staticmethod
    def is_irap(nal_unit_type: int) -> bool:
        return 16 <= nal_unit_type <= 23

    def get_max_poc_lsb(self) -> int:
        return 1 << (self.log2_max_poc_lsb_minus4 + 4)

    def is_ready(self) -> bool:
        return self.sps_parsed and self.pps_parsed

    def skip_profile_tier_level(self, br: BitstreamReader, profile_present_flag: bool, max_num_sub_layers_minus1: int) -> None:
        if profile_present_flag:
            br.skip_bits(2 + 1 + 5 + 32 + 1 + 1 + 1 + 1 + 44)
        br.skip_bits(8)

        sub_layer_profile_present = []
        sub_layer_level_present = []
        for _ in range(max_num_sub_layers_minus1):
            sub_layer_profile_present.append(br.read_bits(1))
            sub_layer_level_present.append(br.read_bits(1))

        if max_num_sub_layers_minus1 > 0:
            for _ in range(max_num_sub_layers_minus1, 8):
                br.skip_bits(2)

        for i in range(max_num_sub_layers_minus1):
            if sub_layer_profile_present[i]:
                br.skip_bits(88)
            if sub_layer_level_present[i]:
                br.skip_bits(8)

    def parse_sps(self, nalu_data: bytes) -> bool:
        rbsp = self.remove_emulation_prevention(nalu_data)
        br = BitstreamReader(rbsp)
        br.skip_bits(16)
        br.skip_bits(4)
        sps_max_sub_layers_minus1 = br.read_bits(3)
        br.skip_bits(1)
        self.skip_profile_tier_level(br, True, sps_max_sub_layers_minus1)
        br.read_ue()
        self.chroma_format_idc = br.read_ue()
        if self.chroma_format_idc == 3:
            self.separate_colour_plane_flag = bool(br.read_bits(1))
        self.pic_width_in_luma_samples = br.read_ue()
        self.pic_height_in_luma_samples = br.read_ue()
        conformance_window_flag = br.read_bits(1)
        if conformance_window_flag:
            br.read_ue(); br.read_ue(); br.read_ue(); br.read_ue()
        br.read_ue()
        br.read_ue()
        self.log2_max_poc_lsb_minus4 = br.read_ue()
        sps_sub_layer_ordering_info_present_flag = br.read_bits(1)
        start_i = 0 if sps_sub_layer_ordering_info_present_flag else sps_max_sub_layers_minus1
        for _ in range(start_i, sps_max_sub_layers_minus1 + 1):
            br.read_ue(); br.read_ue(); br.read_ue()
        self.log2_min_coding_block_size_minus3 = br.read_ue()
        self.log2_diff_max_min_coding_block_size = br.read_ue()
        min_cb_log2 = self.log2_min_coding_block_size_minus3 + 3
        ctb_log2 = min_cb_log2 + self.log2_diff_max_min_coding_block_size
        ctb_size = 1 << ctb_log2
        pic_width_in_ctbs = (self.pic_width_in_luma_samples + ctb_size - 1) // ctb_size
        pic_height_in_ctbs = (self.pic_height_in_luma_samples + ctb_size - 1) // ctb_size
        self.pic_size_in_ctbs_y = pic_width_in_ctbs * pic_height_in_ctbs
        self.slice_segment_address_bits = (self.pic_size_in_ctbs_y - 1).bit_length() if self.pic_size_in_ctbs_y > 1 else 0
        self.sps_parsed = True
        return True

    def parse_pps(self, nalu_data: bytes) -> bool:
        rbsp = self.remove_emulation_prevention(nalu_data)
        br = BitstreamReader(rbsp)
        br.skip_bits(16)
        br.read_ue()
        br.read_ue()
        self.dependent_slice_segments_enabled_flag = bool(br.read_bits(1))
        self.output_flag_present_flag = bool(br.read_bits(1))
        self.num_extra_slice_header_bits = br.read_bits(3)
        self.pps_parsed = True
        return True

    def rewrite_poc(self, nalu_data: bytes, new_poc_lsb: int) -> bytes | None:
        if not self.is_ready() or len(nalu_data) < 3:
            return None
        nal_unit_type = (nalu_data[0] >> 1) & 0x3F
        if not self.is_slice_nalu(nal_unit_type) or self.is_idr(nal_unit_type):
            return None

        rbsp = bytearray(self.remove_emulation_prevention(nalu_data))
        br = BitstreamReader(bytes(rbsp))
        br.skip_bits(16)
        first_slice = bool(br.read_bits(1))
        if self.is_irap(nal_unit_type):
            br.skip_bits(1)
        br.read_ue()
        dependent_slice_segment_flag = False
        if not first_slice:
            if self.dependent_slice_segments_enabled_flag:
                dependent_slice_segment_flag = bool(br.read_bits(1))
            if self.slice_segment_address_bits > 0:
                br.skip_bits(self.slice_segment_address_bits)
        if dependent_slice_segment_flag:
            return None

        br.skip_bits(self.num_extra_slice_header_bits)
        br.read_ue()
        if self.output_flag_present_flag:
            br.skip_bits(1)
        if self.separate_colour_plane_flag:
            br.skip_bits(2)

        poc_bit_offset = br.bit_offset
        poc_bits = self.log2_max_poc_lsb_minus4 + 4
        new_poc_lsb %= self.get_max_poc_lsb()
        for i in range(poc_bits):
            byte_idx = (poc_bit_offset + i) // 8
            bit_idx = 7 - ((poc_bit_offset + i) % 8)
            if byte_idx >= len(rbsp):
                return None
            bit_val = (new_poc_lsb >> (poc_bits - 1 - i)) & 1
            rbsp[byte_idx] = (rbsp[byte_idx] & ~(1 << bit_idx)) | (bit_val << bit_idx)
        return self.add_emulation_prevention(bytes(rbsp))


def split_annexb_nalus(data: bytes) -> list[bytes]:
    starts: list[tuple[int, int]] = []
    i = 0
    size = len(data)
    while i + 3 < size:
        if data[i] == 0 and data[i + 1] == 0:
            if data[i + 2] == 1:
                starts.append((i, 3))
                i += 3
                continue
            if i + 3 < size and data[i + 2] == 0 and data[i + 3] == 1:
                starts.append((i, 4))
                i += 4
                continue
        i += 1
    if not starts:
        return [data] if data else []
    nalus: list[bytes] = []
    for idx, (_, sc_len) in enumerate(starts):
        start_code_pos = starts[idx][0]
        nalu_start = start_code_pos + sc_len
        nalu_end = starts[idx + 1][0] if idx + 1 < len(starts) else size
        if nalu_end > nalu_start:
            nalus.append(data[nalu_start:nalu_end])
    return nalus


@dataclass
class H265StreamRewriter:
    path: Path
    cam_key: str

    def __post_init__(self) -> None:
        self.rewriter = H265PocRewriter()
        self.first_idr_seen = False
        self.poc_counter = 0
        self.skipped_packets = 0
        self.handle = None
        self.buffered_headers: dict[int, bytes] = {}

    def _open(self) -> None:
        if self.handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = open(self.path, 'wb')

    def _write_annexb(self, nalus: list[bytes]) -> None:
        self._open()
        for nalu in nalus:
            self.handle.write(START_CODE)
            self.handle.write(nalu)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def write_payload(self, payload: bytes) -> bool:
        nalus = split_annexb_nalus(payload)
        if not nalus:
            return False

        has_idr = False
        for nalu in nalus:
            if len(nalu) < 2:
                continue
            nal_unit_type = (nalu[0] >> 1) & 0x3F
            if nal_unit_type == 32:
                self.buffered_headers[32] = nalu
            elif nal_unit_type == 33:
                self.buffered_headers[33] = nalu
                self.rewriter.parse_sps(nalu)
            elif nal_unit_type == 34:
                self.buffered_headers[34] = nalu
                self.rewriter.parse_pps(nalu)
            if H265PocRewriter.is_idr(nal_unit_type):
                has_idr = True

        if has_idr:
            if not self.first_idr_seen:
                header_nalus = [self.buffered_headers[k] for k in (32, 33, 34) if k in self.buffered_headers]
                if header_nalus:
                    self._write_annexb(header_nalus)
            self.first_idr_seen = True
            self.poc_counter = 1
            self._write_annexb(nalus)
            return True

        if not self.first_idr_seen:
            self.skipped_packets += 1
            return False

        frame_poc = self.poc_counter % self.rewriter.get_max_poc_lsb() if self.rewriter.is_ready() else 0
        rewritten_nalus: list[bytes] = []
        rewrote_any = False
        for nalu in nalus:
            if len(nalu) < 2:
                rewritten_nalus.append(nalu)
                continue
            rewritten = self.rewriter.rewrite_poc(nalu, frame_poc)
            if rewritten is not None:
                rewritten_nalus.append(rewritten)
                rewrote_any = True
            else:
                rewritten_nalus.append(nalu)
        self._write_annexb(rewritten_nalus)
        if rewrote_any:
            self.poc_counter += 1
        else:
            self.poc_counter += 1
        return True
