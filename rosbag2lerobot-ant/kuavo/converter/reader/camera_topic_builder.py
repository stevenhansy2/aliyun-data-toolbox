"""Camera topic-process mapping builder."""


def build_camera_topic_process_map(
    default_camera_names: list[str],
    camera_topic_specs: dict,
    cam_map: dict,
    topics: list[str],
    use_depth: bool,
    msg_processer,
) -> dict:
    topic_process_map = {}

    for camera in default_camera_names:
        spec = camera_topic_specs.get(camera, {})
        base_topic = cam_map.get(camera, "")

        color_topic = spec.get("color_topic") or (
            f"/{base_topic.split('/')[1][-5:]}/color/image_raw/compressed"
            if base_topic
            else ""
        )
        if color_topic in topics:
            topic_process_map[camera] = {
                "topic": color_topic,
                "msg_process_fn": msg_processer.process_color_image,
            }
            camera_info_topic = spec.get("camera_info_topic") or (
                f"/{base_topic.split('/')[1][-5:]}/color/camera_info"
                if base_topic
                else ""
            )
            if camera_info_topic in topics:
                topic_process_map[f"{camera}_camera_info"] = {
                    "topic": camera_info_topic,
                    "msg_process_fn": msg_processer.process_camera_info,
                }

        if not use_depth:
            continue

        depth_topic_uncompressed = spec.get("depth_uncompressed_topic")
        depth_topic_compressed = spec.get("depth_compressed_topic")
        if not depth_topic_uncompressed or not depth_topic_compressed:
            if "wrist" in camera:
                depth_topic_uncompressed = (
                    f"/{camera[-5:]}/depth/image_rect_raw/compressedDepth"
                )
                depth_topic_compressed = (
                    f"/{camera[-5:]}/depth/image_rect_raw/compressed"
                )
            else:
                depth_topic_uncompressed = f"/{camera[-5:]}/depth/image_raw/compressedDepth"
                depth_topic_compressed = f"/{camera[-5:]}/depth/image_raw/compressed"

        if depth_topic_uncompressed in topics:
            print(f"[INFO] {camera}: 选择未压缩深度话题 {depth_topic_uncompressed}")
            topic_process_map[f"{camera}_depth"] = {
                "topic": depth_topic_uncompressed,
                "msg_process_fn": msg_processer.process_depth_image_16U,
                "fallback_topic": depth_topic_compressed,
                "fallback_fn": msg_processer.process_depth_image,
            }
        elif depth_topic_compressed in topics:
            print(f"[INFO] {camera}: 仅找到压缩深度话题 {depth_topic_compressed}")
            topic_process_map[f"{camera}_depth"] = {
                "topic": depth_topic_compressed,
                "msg_process_fn": msg_processer.process_depth_image,
            }
        else:
            print(f"[WARN] {camera} 未找到深度话题（未压缩或压缩）")

    return topic_process_map

