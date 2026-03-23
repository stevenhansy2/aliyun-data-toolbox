"""Camera topic-process mapping builder for the compat line."""


def build_camera_topic_process_map(
    default_camera_names: list[str],
    camera_topic_specs: dict,
    *,
    use_depth: bool,
    msg_processer,
) -> dict:
    topic_process_map = {}

    for camera in default_camera_names:
        spec = camera_topic_specs.get(camera, {})

        color_topic = spec.get("color_topic")
        color_candidates = spec.get("color_topic_candidates") or ([color_topic] if color_topic else [])
        depth_topic = spec.get("depth_topic")
        depth_candidates = spec.get("depth_topic_candidates") or ([depth_topic] if depth_topic else [])
        camera_info_topic = spec.get("camera_info_topic")

        if camera.startswith("depth_"):
            if use_depth and depth_topic:
                topic_process_map[camera] = {
                    "topic": depth_topic,
                    "topic_candidates": depth_candidates,
                    "msg_process_fn": msg_processer.process_depth_image,
                }
            continue

        if color_topic:
            topic_process_map[camera] = {
                "topic": color_topic,
                "topic_candidates": color_candidates,
                "msg_process_fn": msg_processer.process_color_image,
            }

        if camera_info_topic:
            topic_process_map[f"{camera}_camera_info"] = {
                "topic": camera_info_topic,
                "topic_candidates": [camera_info_topic, f"{camera_info_topic}/"],
                "msg_process_fn": msg_processer.process_camera_info,
            }

    return topic_process_map
