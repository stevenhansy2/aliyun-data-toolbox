"""Camera parameter and image extraction helpers."""

import json
import os

import numpy as np

def save_camera_extrinsic_params(cameras, output_dir):
    """
    为每个相机生成外参文件

    Args:
        cameras: 相机名称列表，如 ['head_cam_h', 'wrist_cam_r', 'wrist_cam_l']
        output_dir: 输出目录路径
    """
    import os
    import json

    os.makedirs(output_dir, exist_ok=True)

    # 预定义的外参数据
    extrinsic_data = {
        "camera_top": {
            "rotation_matrix": [
                [0.8829475928589267, 0.0, 0.4694715627858914],
                [0.0, 1.0, 0.0],
                [-0.4694715627858914, 0.0, 0.8829475928589267],
            ],
            "translation_vector": [
                0.0967509784707853,
                0.0175003248712456,
                0.12595326511272098,
            ],
        },
        "camera_wrist_right": {
            "rotation_matrix": [
                [-0.7071096173630955, -0.7071039221275017, 0.0001798458248664092],
                [0.1830017433973951, -0.18275752957580388, 0.9659762146641413],
                [-0.6830127018922347, 0.6830839836325056, 0.25863085732104],
            ],
            "translation_vector": [
                0.115405591590931,
                0.015431235212043481,
                -0.10772412843089599,
            ],
        },
        "camera_wrist_left": {
            "rotation_matrix": [
                [-0.7076167402785503, 0.5411120873595152, 0.45439658646493686],
                [-0.17642866083771364, 0.48740501040759066, -0.855166231480516],
                [-0.6842159575109087, -0.685298522355772, -0.249428263724113],
            ],
            "translation_vector": [
                0.11540559159102,
                -0.014558611066996074,
                -0.110123491595499,
            ],
        },
    }

    for camera in cameras:
        if camera in extrinsic_data:
            extrinsic_json = {"extrinsic": extrinsic_data[camera]}

            json_path = os.path.join(output_dir, f"{camera}_extrinsic.json")
            with open(json_path, "w") as f:
                json.dump(extrinsic_json, f, indent=4)
            print(f"Saved {json_path}")
        else:
            print(f"Warning: No extrinsic data found for camera {camera}")


def load_raw_images_per_camera(
    bag_data: dict, default_camera_names: list[str]
) -> dict[str, list]:
    imgs_per_cam = {}
    for camera in default_camera_names:
        imgs_per_cam[camera] = [msg["data"] for msg in bag_data[camera]]
    return imgs_per_cam


def load_raw_depth_images_per_camera(bag_data: dict, default_camera_names: list[str]):
    imgs_per_cam = {}
    compressed_per_cam = {}
    for camera in default_camera_names:
        key = f"{camera}_depth"
        imgs_per_cam[key] = [msg["data"] for msg in bag_data[key]]
        # 只取第一帧的压缩状态（假设所有帧一致）
        if bag_data[key]:
            compressed_per_cam[key] = bag_data[key][0].get("compressed", None)
        else:
            compressed_per_cam[key] = None
    print("+" * 20, compressed_per_cam)
    return imgs_per_cam, compressed_per_cam


def load_camera_info_per_camera(
    bag_data: dict, default_camera_names: list[str]
) -> dict:
    info_per_cam = {}
    distortion_model = {}
    for camera in default_camera_names:
        info_per_cam[camera] = np.array(
            [msg["data"] for msg in bag_data[f"{camera}_camera_info"]], dtype=np.float32
        )
        distortion_model[camera] = [
            msg["distortion_model"] for msg in bag_data[f"{camera}_camera_info"]
        ]

    return info_per_cam, distortion_model


def save_camera_info_to_json(info_per_cam, output_dir):
    """
    将 info_per_cam 中的数据还原为每个摄像头的 json 文件，包含 D、K、R、P 四个参数
    """
    os.makedirs(output_dir, exist_ok=True)
    for camera, cam_infos in info_per_cam.items():
        # 取第0帧的参数（通常所有帧的内参都一样）
        camera_vec = cam_infos[0]
        # 假设 D5, K9, R9, P12 顺序拼接
        D = camera_vec[0:5].tolist()
        K = camera_vec[5:14].tolist()
        R = camera_vec[14:23].tolist()
        P = camera_vec[23:35].tolist()
        params = {"D": D, "K": K, "R": R, "P": P}
        # 生成文件名
        json_path = os.path.join(output_dir, f"{camera}_intrinsic_params.json")
        with open(json_path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Saved {json_path}")


def save_camera_info_to_json_new(info_per_cam, distortion_model, output_dir):
    """
    将 info_per_cam 中的数据转换为 intrinsic 格式并保存为 json
    支持 D5/K9/R9/P12 或 D8/K9/R9/P12 等不同格式
    """
    os.makedirs(output_dir, exist_ok=True)
    for camera, cam_infos in info_per_cam.items():
        if cam_infos is None or len(cam_infos) == 0:
            print(f"[WARN] {camera} 无相机内参数据，跳过保存")
            continue
        camera_vec = cam_infos[0]
        total_len = len(camera_vec)
        # 动态判断 D/K/R/P 长度
        # 常见有 D5/K9/R9/P12（总35），D8/K9/R9/P12（总38）
        if total_len == 35:
            D_len, K_len, R_len, P_len = 5, 9, 9, 12
        elif total_len == 38:
            D_len, K_len, R_len, P_len = 8, 9, 9, 12
        else:
            raise ValueError(f"未知的camera_vec长度: {total_len}，请检查相机参数格式")
        D = camera_vec[0:D_len].tolist()
        K = camera_vec[D_len : D_len + K_len].tolist()
        R = camera_vec[D_len + K_len : D_len + K_len + R_len].tolist()
        P = camera_vec[D_len + K_len + R_len : D_len + K_len + R_len + P_len].tolist()
        # distortion_model字段（优先取第一个，如有多个可自行调整）
        if camera in distortion_model and distortion_model[camera]:
            model = getattr(distortion_model[camera][0], "distortion_model", None)
            if model is None:
                model = (
                    distortion_model[camera][0]
                    if isinstance(distortion_model[camera][0], str)
                    else "unknown"
                )
        else:
            model = "unknown"
        # intrinsic格式
        intrinsic = {
            "fx": K[0],  # K[0]
            "fy": K[4],  # K[4]
            "ppx": K[2],  # K[2]
            "ppy": K[5],  # K[5]
            "distortion_model": model,
            "k1": D[0] if len(D) > 0 else None,
            "k2": D[1] if len(D) > 1 else None,
            "k3": D[4] if len(D) > 4 else None,
            "p1": D[2] if len(D) > 2 else None,
            "p2": D[3] if len(D) > 3 else None,
        }
        json_path = os.path.join(output_dir, f"{camera}_intrinsic.json")
        with open(json_path, "w") as f:
            json.dump({"intrinsic": intrinsic}, f, indent=2)
        print(f"Saved {json_path}")


