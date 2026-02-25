"""Compatibility facade for utility functions used by conversion pipeline."""

from converter.utils.camera_utils import (
    get_mapped_camera_name,
    get_mapped_filename,
    load_camera_info_per_camera,
    load_raw_depth_images_per_camera,
    load_raw_images_per_camera,
    recursive_filter_and_position,
    save_camera_extrinsic_params,
    save_camera_info_to_json,
    save_camera_info_to_json_new,
)
from converter.utils.data_quality import (
    analyze_stillness_frames,
    detect_and_trim_bag_data,
    detect_stillness_from_image_data,
    flip_camera_arrays_if_needed,
    should_flip_camera,
    swap_left_right_data_if_needed,
    trim_all_bag_data_by_frames,
    validate_episode_data_consistency,
)
from converter.utils.hdf5_utils import (
    IncrementalHDF5Writer,
    write_dict_to_hdf5_batched,
)
from converter.utils.video_parallel import (
    save_color_videos_ffmpeg_parallel,
    save_color_videos_parallel,
    save_depth_videos_16U_parallel,
    save_depth_videos_enhanced_parallel,
    save_depth_videos_ffmpeg_parallel,
    save_depth_videos_parallel,
)
from converter.utils.video_streaming import (
    StreamingVideoWriter,
    save_color_videos_streaming,
    save_depth_videos_16U_streaming,
)

__all__ = [
    "IncrementalHDF5Writer",
    "StreamingVideoWriter",
    "analyze_stillness_frames",
    "detect_and_trim_bag_data",
    "detect_stillness_from_image_data",
    "flip_camera_arrays_if_needed",
    "get_mapped_camera_name",
    "get_mapped_filename",
    "load_camera_info_per_camera",
    "load_raw_depth_images_per_camera",
    "load_raw_images_per_camera",
    "recursive_filter_and_position",
    "save_camera_extrinsic_params",
    "save_camera_info_to_json",
    "save_camera_info_to_json_new",
    "save_color_videos_ffmpeg_parallel",
    "save_color_videos_parallel",
    "save_color_videos_streaming",
    "save_depth_videos_16U_parallel",
    "save_depth_videos_16U_streaming",
    "save_depth_videos_enhanced_parallel",
    "save_depth_videos_ffmpeg_parallel",
    "save_depth_videos_parallel",
    "should_flip_camera",
    "swap_left_right_data_if_needed",
    "trim_all_bag_data_by_frames",
    "validate_episode_data_consistency",
    "write_dict_to_hdf5_batched",
]
