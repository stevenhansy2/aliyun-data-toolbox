"""Post-processing utilities for torque/current conversion and HDF5 export."""

import os

import numpy as np

class PostProcessorUtils:
    @staticmethod
    def torque_to_current_batch(
        torque_data: np.ndarray,
        MOTOR_C2T=[
            2,
            1.05,
            1.05,
            2,
            2.1,
            2.1,
            2,
            1.05,
            1.05,
            2,
            2.1,
            2.1,
            1.05,
            5,
            2.3,
            5,
            4.7,
            4.7,
            4.7,
            1.05,
            5,
            2.3,
            5,
            4.7,
            4.7,
            4.7,
            0.21,
            4.7,
        ],
    ):
        """
        将扭矩数据批量转换为电流数据

        Args:
            torque_data: 扭矩数据数组(N, M)
            MOTOR_C2T: 电流转扭矩系数数组，默认值为 kuavo-ros-control 中定义的系数
        Returns:
            电流数据数组
        """
        if torque_data.shape[1] != len(MOTOR_C2T):
            print(
                f"警告: 扭矩数据长度({torque_data.shape[1]})与C2T系数数量({len(MOTOR_C2T)})不匹配"
            )
            return None

        # 复制数据避免修改原始数据
        current_data = torque_data.copy()

        from itertools import chain

        # 13~18 为左臂ruiwo电机数据, 20~27 为右臂ruiwo电机数据
        # 对于这些电机需要先除以MOTOR_C2T系数再乘以2.1
        for i in chain(range(13, 19), range(20, 28)):  # 修正为27+1=28
            current_data[:, i] = (torque_data[:, i] / MOTOR_C2T[i]) * 2.1

        # 1, 2, 7, 8, 12, 19 号电机需要特殊处理
        for i in [1, 2, 7, 8, 12, 19]:
            current_data[:, i] = (torque_data[:, i] / MOTOR_C2T[i]) * 1.2

        # 其他电机：EC电机，直接除以MOTOR_C2T系数
        other_indices = [
            i
            for i in range(len(MOTOR_C2T))
            if i not in chain(range(13, 19), range(20, 28), [1, 2, 7, 8, 12, 19])
        ]
        for i in other_indices:
            current_data[:, i] = torque_data[:, i] / MOTOR_C2T[i]

        return current_data

    @staticmethod
    def current_to_torque(
        current_data: np.ndarray,
        MOTOR_C2T=[
            2,
            1.05,
            1.05,
            2,
            2.1,
            2.1,
            2,
            1.05,
            1.05,
            2,
            2.1,
            2.1,
            1.05,
            5,
            2.3,
            5,
            4.7,
            4.7,
            4.7,
            1.05,
            5,
            2.3,
            5,
            4.7,
            4.7,
            4.7,
            0.21,
            4.7,
        ],
    ):
        """
        将 sensors_data_raw 中的 joint_torque 电流数据转换为扭矩数据

        Args:
            current_data: 电流数据数组(N, 28)
            MOTOR_C2T: 电流转扭矩系数数组，默认值为 kuavo-ros-control 中定义的系数
        Returns:
            扭矩数据数组
        """
        if len(current_data) != len(MOTOR_C2T):
            print(
                f"警告: 电流数据长度({len(current_data)})与C2T系数数量({len(MOTOR_C2T)})不匹配"
            )
            # 扩展或截断系数数组
            return None

        torque_data = []
        # "MOTORS_TYPE":[
        # "PA100_18", "PA100", "PA100", "PA100_18", "CK", "CK",
        # "PA100_18", "PA100", "PA100", "PA100_18", "CK", "CK",
        # "PA100", "ruiwo", "ruiwo", "ruiwo", "ruiwo", "ruiwo", "ruiwo",
        # "PA100", "ruiwo", "ruiwo", "ruiwo", "ruiwo", "ruiwo", "ruiwo", "ruiwo", "ruiwo"],

        for i, current in enumerate(current_data):
            # kuavo-ros-control/src/kuavo_common/include/kuavo_common/common/kuavo_settings.h
            # 中定义了 ruiwo 电机电流转扭矩系数 CK_C2T = 2.1，所以这里除以 2.1 转化回原始电流

            # 13~18 为左臂ruiwo电机数据, 20~25 为右臂ruiwo电机数据
            # 对于这些电机需要先除以2.1转换回原始电流
            if 13 <= i <= 18 or 20 <= i <= 27:
                torque = (current / 2.1) * MOTOR_C2T[i]
            elif i == 1 or i == 2 or i == 7 or i == 8 or i == 12 or i == 19:
                torque = (current / 1.2) * MOTOR_C2T[i]
            else:

                # EC 电机 sensors_data_raw 中已经是扭矩值
                torque = current
            torque_data.append(torque)

        return np.array(torque_data)

    @staticmethod
    def current_to_torque_batch(
        current_data: np.ndarray,
        MOTOR_C2T=[
            2,
            1.05,
            1.05,
            2,
            2.1,
            2.1,
            2,
            1.05,
            1.05,
            2,
            2.1,
            2.1,
            1.05,
            5,
            2.3,
            5,
            4.7,
            4.7,
            4.7,
            1.05,
            5,
            2.3,
            5,
            4.7,
            4.7,
            4.7,
            0.21,
            4.7,
        ],
    ):
        """
        将 sensors_data_raw 中的 joint_torque 电流数据转换为扭矩数据

        Args:
            current_data: 电流数据数组(N, M)
            MOTOR_C2T: 电流转扭矩系数数组，默认值为 kuavo-ros-control 中定义的系数
        Returns:
            扭矩数据数组
        """

        if current_data.shape[1] != len(MOTOR_C2T):
            print(
                f"警告: 电流数据长度({current_data.shape[1]})与C2T系数数量({len(MOTOR_C2T)})不匹配"
            )
            # 扩展或截断系数数组
            return None

        from itertools import chain

        for i in chain(range(13, 19), range(20, 28)):
            current_data[:, i] = current_data[:, i] / 2.1 * MOTOR_C2T[i]
        for i in [1, 2, 7, 8, 12, 19]:
            current_data[:, i] = current_data[:, i] / 1.2 * MOTOR_C2T[i]
        # 对于其他电机直接使用原始电流
        # EC 电机 sensors_data_raw 中已经是扭矩值
        return current_data

    @staticmethod
    def save_to_hdf5(low_dim_data, file_path):
        """将数据保存为符合库帕思通用版数据格式的HDF5文件"""
        import h5py

        # 确保输出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        def create_datasets_recursively(group, data_dict, current_path=""):
            """递归创建数据集和组"""
            for key, value in data_dict.items():
                full_path = f"{current_path}/{key}" if current_path else key

                if isinstance(value, dict):
                    # 如果是字典，创建子组并递归处理
                    subgroup = group.create_group(key)
                    create_datasets_recursively(subgroup, value, full_path)
                else:
                    # 如果是数据，创建数据集
                    try:
                        # 处理不同类型的数据
                        if isinstance(value, (list, tuple)):
                            value = np.array(value)

                        # 根据数据类型和路径进行特殊处理
                        processed_value = process_data_by_path(value, full_path)

                        # 创建数据集
                        group.create_dataset(key, data=processed_value)
                        print(
                            f"创建数据集: {full_path}, 形状: {processed_value.shape}, 类型: {processed_value.dtype}"
                        )

                    except Exception as e:
                        print(f"警告: 无法创建数据集 {full_path}: {e}")
                        # 创建空数据集作为占位符
                        try:
                            empty_data = np.array([])
                            group.create_dataset(key, data=empty_data)
                        except:
                            pass

        def process_data_by_path(value, path):
            """根据数据路径对数据进行特殊处理"""
            # 时间戳处理 - 扩展识别新的时间戳字段
            timestamp_fields = [
                "timestamps",
                "head_color_mp4_camera_timestamps",
                "hand_left_color_mp4_timestamps",
                "hand_right_color_mp4_timestamps",
                "head_depth_mkv_camera_timestamps",
                "hand_left_depth_mkv_timestamps",
                "hand_right_depth_mkv_timestamps",
                "camera_extrinsics_timestamps",
                "head_timestamps" "joint_timestamps",
                "effector_dexhand_timestamps",
                "effector_lejuclaw_timestamps",
            ]

            if any(ts_field in path for ts_field in timestamp_fields):
                if value.dtype != np.int64:
                    # 转换时间戳为纳秒级整数
                    if np.issubdtype(value.dtype, np.floating):
                        return (value * 1e9).astype(np.int64)
                    else:
                        return value.astype(np.int64)
                return value

            # 索引数据处理
            elif "index" in path:
                return value.astype(np.int64)

            # 其他数值数据处理
            elif np.issubdtype(value.dtype, np.number):
                # 根据数据类型决定精度
                if np.issubdtype(value.dtype, np.integer):
                    return value.astype(np.int32)
                else:
                    return value.astype(np.float32)

            # 保持原始数据类型
            return value

        def add_missing_required_fields(f, low_dim_data):
            """添加库帕思格式中必需但缺失的字段，使用null机制"""

            # 获取时间戳长度作为参考
            if "timestamps" in low_dim_data:
                N = len(low_dim_data["timestamps"])
            else:
                N = 1000  # 默认值
                for key, value in low_dim_data.items():
                    if hasattr(value, "__len__") and not isinstance(value, str):
                        N = len(value)
                        break

            # 创建控制索引
            control_indices = np.arange(N, dtype=np.int64)

            def create_null_dataset(group, name, shape, dtype):
                """创建一个表示缺失数据的数据集"""
                # 方法1: 使用NaN表示缺失数据（仅适用于浮点数）
                if dtype == np.float32或dtype == np.float64:
                    data = np.full(shape, np.nan, dtype=dtype)
                    dataset = group.create_dataset(name, data=data)
                    # 添加属性标记这是缺失数据
                    dataset.attrs["missing_data"] = True
                    dataset.attrs["description"] = f"Missing data filled with NaN"
                    return dataset

                # 方法2: 创建空数据集（对于整数类型）
                elif np.issubdtype(dtype, np.integer):
                    # 对于整数，使用最小值表示缺失
                    if dtype == np.int32:
                        fill_value = np.iinfo(np.int32).min
                    elif dtype == np.int64:
                        fill_value = np.iinfo(np.int64).min
                    else:
                        fill_value = -999999  # 默认缺失值

                    data = np.full(shape, fill_value, dtype=dtype)
                    dataset = group.create_dataset(name, data=data)
                    dataset.attrs["missing_data"] = True
                    dataset.attrs["fill_value"] = fill_value
                    dataset.attrs["description"] = (
                        f"Missing data filled with {fill_value}"
                    )
                    return dataset

                # 方法3: 不创建数据集，仅添加占位符属性
                else:
                    # 创建一个只有属性的组来表示缺失
                    missing_group = group.create_group(name + "_missing")
                    missing_group.attrs["missing_data"] = True
                    missing_group.attrs["expected_shape"] = shape
                    missing_group.attrs["expected_dtype"] = str(dtype)
                    missing_group.attrs["description"] = (
                        "Data not available - missing field"
                    )
                    return missing_group

            def create_optional_dataset(
                group, name, shape, dtype, description="Optional field not available"
            ):
                """创建可选的数据集，明确标记为不可用"""
                # 方法4: 创建虚拟数据集，长度为0
                empty_data = np.array([], dtype=dtype)
                dataset = group.create_dataset(name, data=empty_data, maxshape=shape)
                dataset.attrs["data_available"] = False
                dataset.attrs["expected_shape"] = shape
                dataset.attrs["description"] = description
                return dataset

            # 检查并添加缺失的 action 组字段
            if "action" in f:
                action_group = f["action"]

                # # 添加缺失的 robot 组
                # if "robot" not in action_group:
                #     robot_group = action_group.create_group("robot")
                #     create_null_dataset(robot_group, "velocity", (N, 2), np.float32)
                #     create_null_dataset(robot_group, "index", (N,), np.float32)
                #     print(f"添加缺失字段: action/robot (使用NaN表示缺失)")

                # # 添加缺失的 waist 组
                # if "waist" not in action_group:
                #     waist_group = action_group.create_group("waist")
                #     create_null_dataset(waist_group, "position", (N, 2), np.float32)
                #     create_null_dataset(waist_group, "index", (N,), np.float32)
                #     print(f"添加缺失字段: action/waist (使用NaN表示缺失)")

                # # 添加缺失的 end 组
                # if "end" not in action_group:
                #     end_group = action_group.create_group("end")
                #     create_null_dataset(end_group, "orientation", (N, 2, 4), np.float32)
                #     create_null_dataset(end_group, "position", (N, 2, 3), np.float32)
                #     create_null_dataset(end_group, "index", (N,), np.float32)
                #     print(f"添加缺失字段: action/end (使用NaN表示缺失)")

            # 检查并添加缺失的 state 组字段
            if "state" in f:
                state_group = f["state"]

                # # 添加缺失的 end 组
                # if "end" not in state_group:
                #     end_group = state_group.create_group("end")
                #     create_null_dataset(end_group, "angular", (N, 2, 3), np.float32)
                #     create_null_dataset(end_group, "orientation", (N, 2, 4), np.float32)
                #     create_null_dataset(end_group, "position", (N, 2, 3), np.float32)
                #     create_null_dataset(end_group, "velocity", (N, 2, 3), np.float32)
                #     create_null_dataset(end_group, "wrench", (N, 2, 6), np.float32)
                #     print(f"添加缺失字段: state/end (使用NaN表示缺失)")

                # 添加缺失的 robot 组
                if "robot" not in state_group:
                    robot_group = state_group.create_group("robot")

                    # 对于机器人姿态，如果没有IMU数据，明确标记为缺失
                    if "imu" in low_dim_data and "quat_xyzw" in low_dim_data["imu"]:
                        imu_data_quat_xyzw = low_dim_data["imu"]["quat_xyzw"]
                        if (
                            hasattr(imu_data_quat_xyzw, "shape")
                            and len(imu_data_quat_xyzw.shape) > 1
                            and imu_data_quat_xyzw.shape[1] >= 4
                        ):
                            # 有IMU数据，直接使用
                            orientation = np.zeros((N, 4), dtype=np.float32)
                            orientation[:, :] = imu_data_quat_xyzw
                            dataset = robot_group.create_dataset(
                                "orientation", data=orientation
                            )
                            dataset.attrs["data_source"] = "IMU sensor"
                            dataset.attrs["missing_data"] = False
                            print(f"从IMU数据提取机器人姿态")
                        else:
                            # IMU数据格式不对，标记为缺失
                            create_null_dataset(
                                robot_group, "orientation", (N, 4), np.float32
                            )
                            print(f"IMU数据格式异常，姿态数据标记为缺失")
                    else:
                        # 没有IMU数据，标记为缺失
                        create_null_dataset(
                            robot_group, "orientation", (N, 4), np.float32
                        )
                        print(f"无IMU数据，姿态数据标记为缺失")

                    # 其他机器人状态标记为缺失
                    # create_null_dataset(robot_group, "orientation_drift", (N, 4), np.float32)
                    # create_null_dataset(robot_group, "position", (N, 3), np.float32)
                    # create_null_dataset(robot_group, "position_drift", (N, 3), np.float32)
                    print(f"添加缺失字段: state/robot (使用NaN/缺失值表示)")

                # # 添加缺失的 waist 组
                # if "waist" not in state_group:
                #     waist_group = state_group.create_group("waist")
                #     create_null_dataset(waist_group, "effort", (N, 2), np.float32)
                #     create_null_dataset(waist_group, "position", (N, 2), np.float32)
                #     create_null_dataset(waist_group, "velocity", (N, 2), np.float32)
                #     print(f"添加缺失字段: state/waist (使用NaN表示缺失)")

                # 为现有组添加缺失的数据集
                # if "effector" in state_group:
                #     effector_group = state_group["effector"]
                #     if "force" not in effector_group:
                #         create_null_dataset(effector_group, "force", (N, 2), np.float32)
                #         print(f"添加缺失字段: state/effector/force (使用NaN表示缺失)")

                if "head" in state_group:
                    head_group = state_group["head"]
                    if "effort" not in head_group:
                        create_null_dataset(head_group, "effort", (N, 2), np.float32)
                        print(f"添加缺失字段: state/head/effort (使用NaN表示缺失)")

                if "joint" in state_group:
                    joint_group = state_group["joint"]
                    # 获取关节数量
                    joint_count = 14  # 默认值
                    if "position" in joint_group:
                        joint_count = joint_group["position"].shape[1]
                    elif "velocity" in joint_group:
                        joint_count = joint_group["velocity"].shape[1]

                    if "current_value" not in joint_group:
                        create_null_dataset(
                            joint_group, "current_value", (N, joint_count), np.float32
                        )
                        print(
                            f"添加缺失字段: state/joint/current_value (使用NaN表示缺失)"
                        )

                    if "effort" not in joint_group:
                        create_null_dataset(
                            joint_group, "effort", (N, joint_count), np.float32
                        )
                        print(f"添加缺失字段: state/joint/effort (使用NaN表示缺失)")

            # 添加 other_sensors 组（标记为可选）
            # if "other_sensors" not in f:
            #     other_group = f.create_group("other_sensors")
            #     other_group.attrs['description'] = 'Optional sensor data - currently empty'
            #     other_group.attrs['data_available'] = False
            #     print(f"添加缺失字段: other_sensors (标记为可选数据)")
            # 新增：在根级别添加时间戳字段的存在性信息

        # 创建 HDF5 文件
        with h5py.File(file_path, "w") as f:
            print(f"开始创建HDF5文件: {file_path}")

            # 递归创建所有数据集和组
            create_datasets_recursively(f, low_dim_data)

            # 添加库帕思格式要求的缺失字段填充为NaN或缺失值
            add_missing_required_fields(f, low_dim_data)

        print(f"数据已成功保存为HDF5格式: {file_path}")
        return file_path


if __name__ == "__main__":
    # 创建测试实例
    print("创建测试实例...")

    # 模拟配置
    class TestConfig:
        def __init__(self):
            self.default_camera_names = ["head_cam_h"]
            self.train_hz = 30
            self.main_timeline_fps = 30
            self.sample_drop = 0
            self.resize = type("obj", (object,), {"width": 640, "height": 480})()
            self.topics = []
            self.eef_type = "dexhand"

    config = TestConfig()
    reader = KuavoRosbagReader(config)

    # 创建测试数据：模拟前置步骤处理后的数据特征
    # - 所有间隔都小于40ms
    # - 总帧率约32Hz（需要删除帧降到30Hz）
    print("创建测试数据...")

    # 生成32Hz的基本时间戳序列
    base_interval = 1.0 / 32.0  # 32Hz = 31.25ms间隔
    total_frames = 800  # 足够长的数据
    total_duration = total_frames * base_interval  # 总时长

    # 创建均匀的32Hz时间戳作为基础
    uniform_timestamps = np.linspace(1.0, 1.0 + total_duration, total_frames)

    # 添加一些随机性，但确保间隔始终<40ms
    timestamps = []
    for i in range(total_frames):
        base_ts = uniform_timestamps[i]

        if i == 0:
            # 第一帧保持不变
            timestamps.append(base_ts)
        else:
            # 添加随机偏移，但确保与前一帧的间隔在15-38ms之间
            prev_ts = timestamps[-1]
            min_interval = 0.015  # 15ms
            max_interval = 0.038  # 38ms

            # 计算理想的下一个时间戳
            ideal_next = prev_ts + base_interval

            # 添加随机偏移，但限制在安全范围内
            random_offset = np.random.uniform(-0.008, 0.008)  # ±8ms随机偏移
            candidate_ts = ideal_next + random_offset

            # 确保间隔在安全范围内
            actual_interval = candidate_ts - prev_ts
            if actual_interval < min_interval:
                candidate_ts = prev_ts + min_interval
            elif actual_interval > max_interval:
                candidate_ts = prev_ts + max_interval

            timestamps.append(candidate_ts)

    main_timestamps = np.array(timestamps)

    # 验证生成的时间戳质量
    intervals_ms = np.diff(main_timestamps) * 1000
    max_interval_ms = np.max(intervals_ms)
    min_interval_ms = np.min(intervals_ms)
    avg_interval_ms = np.mean(intervals_ms)

    # 确保所有间隔都小于40ms
    assert (
        max_interval_ms < 40.0
    ), f"生成的最大间隔 {max_interval_ms:.1f}ms 超过40ms限制"

    # 子时间戳：比主时间戳晚2ms
    child_timestamps = main_timestamps + 0.002

    # 创建对应的数据
    valid_modalities = {
        "head_cam_h": [
            {"timestamp": ts, "data": f"main_frame_{i}", "frame_id": i}
            for i, ts in enumerate(main_timestamps)
        ],
        "child_sensor": [
            {"timestamp": ts, "data": f"child_data_{i}", "sensor_value": i * 10}
            for i, ts in enumerate(child_timestamps)
        ],
    }

    # 计算初始帧率
    time_span = main_timestamps[-1] - main_timestamps[0]
    initial_fps = len(main_timestamps) / time_span
    target_fps = 30.095

    print(f"测试数据创建完成:")
    print(f"  主时间戳长度: {len(main_timestamps)}")
    print(f"  时间跨度: {time_span:.3f}s")
    print(f"  初始帧率: {initial_fps:.2f}Hz")
    print(f"  目标帧率: {target_fps:.2f}Hz")
    print(f"  需要删除约 {len(main_timestamps) - int(time_span * target_fps)} 帧")

    print(f"  时间间隔统计:")
    print(f"    平均间隔: {avg_interval_ms:.1f}ms")
    print(f"    最大间隔: {max_interval_ms:.1f}ms")
    print(f"    最小间隔: {min_interval_ms:.1f}ms")
    print(f"    ✓ 所有间隔都在40ms以内（模拟前置处理完成）")

    # 验证帧率合理性
    if 31.5 <= initial_fps <= 33.0:
        print(f"    ✓ 初始帧率 {initial_fps:.2f}Hz 在期望范围内（31.5-33Hz）")
    else:
        print(f"    ⚠️ 初始帧率 {initial_fps:.2f}Hz 不在期望范围内")

    print("\n初始数据样本（前10帧）:")
    print("主时间戳:")
    for i in range(min(10, len(main_timestamps))):
        interval_ms = 0
        if i > 0:
            interval_ms = (main_timestamps[i] - main_timestamps[i - 1]) * 1000
        print(f"  帧{i}: {main_timestamps[i]:.6f}s (间隔: {interval_ms:.1f}ms)")

    print("\n子时间戳样本（前5帧）:")
    for i in range(min(5, len(valid_modalities["child_sensor"]))):
        item = valid_modalities["child_sensor"][i]
        main_ts = main_timestamps[i]
        diff_ms = (item["timestamp"] - main_ts) * 1000
        print(
            f"  帧{i}: {item['timestamp']:.6f}s (与主时间戳差: {diff_ms:.1f}ms, 数据: {item['data']})"
        )

    print("\n开始测试 _remove_frames_to_decrease_fps...")
    print("=" * 60)

    try:
        # 调用函数进行测试
        result_timestamps, result_modalities = reader._remove_frames_to_decrease_fps(
            main_timestamps.copy(),  # 使用副本避免修改原数据
            {k: list(v) for k, v in valid_modalities.items()},  # 深拷贝
            target_fps,
            time_span,
        )

        print("=" * 60)
        print("测试完成!")

        # 验证结果
        final_time_span = result_timestamps[-1] - result_timestamps[0]
        final_fps = len(result_timestamps) / final_time_span

        print(f"\n结果统计:")
        print(f"  最终时间戳长度: {len(result_timestamps)}")
        print(f"  最终时间跨度: {final_time_span:.3f}s")
        print(f"  最终帧率: {final_fps:.3f}Hz")
        print(f"  删除帧数: {len(main_timestamps) - len(result_timestamps)}")

        # 验证最终时间戳质量
        if len(result_timestamps) > 1:
            final_intervals_ms = np.diff(result_timestamps) * 1000
            max_final_interval = np.max(final_intervals_ms)
            avg_final_interval = np.mean(final_intervals_ms)
            std_final_interval = np.std(final_intervals_ms)

            print(f"\n最终时间戳质量:")
            print(f"  最大间隔: {max_final_interval:.1f}ms")
            print(f"  平均间隔: {avg_final_interval:.1f}ms")
            print(f"  间隔标准差: {std_final_interval:.1f}ms")

            if max_final_interval <= 40:
                print(f"  ✓ 所有间隔都在40ms以内")
            else:
                large_final_intervals = np.sum(final_intervals_ms > 40)
                print(f"  ❌ 仍有 {large_final_intervals} 个间隔超过40ms")

        # 验证子时间戳同步性（抽样检查）
        print(f"\n子时间戳同步性验证（抽样检查前20帧）:")
        sync_errors = 0
        check_frames = min(
            20, len(result_timestamps), len(result_modalities["child_sensor"])
        )

        for i in range(check_frames):
            main_ts = result_timestamps[i]
            child_item = result_modalities["child_sensor"][i]
            child_ts = child_item["timestamp"]
            expected_diff = 0.002  # 原始2ms差值
            actual_diff = child_ts - main_ts
            diff_error = abs(actual_diff - expected_diff) * 1000

            if diff_error > 0.1:  # 0.1ms容差
                sync_errors += 1
                print(f"  帧{i}: 同步偏差 {diff_error:.3f}ms")

                # 检查是否是重新平均过的帧
                reaveraged = child_item.get("timestamp_reaveraged", False)
                if reaveraged:
                    delta = child_item.get("timestamp_delta", 0)
                    print(f"        (该帧已重新平均, delta: {delta:.6f}s)")

        if sync_errors == 0:
            print(f"  ✓ 抽样检查的 {check_frames} 帧都保持了2ms的相对关系")
        else:
            print(f"  ❌ 在 {check_frames} 帧中发现 {sync_errors} 个同步偏差")

        # 显示处理前后的对比
        print(f"\n处理前后对比:")
        print(
            f"  长度: {len(main_timestamps)} -> {len(result_timestamps)} (-{len(main_timestamps) - len(result_timestamps)})"
        )
        print(f"  帧率: {initial_fps:.2f}Hz -> {final_fps:.2f}Hz")
        print(f"  最大间隔: {max_interval_ms:.1f}ms -> {max_final_interval:.1f}ms")

        # 测试结论
        length_ok = len(result_timestamps) >= 300  # 最终长度足够
        fps_ok = final_fps <= target_fps  # 帧率达标
        interval_ok = max_final_interval <= 40  # 间隔达标
        sync_ok = sync_errors == 0  # 同步达标

        print(f"\n测试结论:")
        print(
            f"  长度检查: {'✅' if length_ok else '❌'} ({len(result_timestamps)} >= 300)"
        )
        print(
            f"  帧率检查: {'✅' if fps_ok else '❌'} ({final_fps:.2f} <= {target_fps})"
        )
        print(
            f"  间隔检查: {'✅' if interval_ok else '❌'} ({max_final_interval:.1f} <= 40ms)"
        )
        print(f"  同步检查: {'✅' if sync_ok else '❌'}")

        if length_ok and fps_ok and interval_ok and sync_ok:
            print(f"  🎉 所有测试通过！滑动窗口删除+重新平均算法工作正常")
        else:
            print(f"  ⚠️ 部分测试未通过，需要进一步优化")

        # 额外检查：验证删除+重新平均的效果
        print(f"\n重新平均效果验证:")
        reaveraged_count = 0
        for item in result_modalities["child_sensor"]:
            if item.get("timestamp_reaveraged", False):
                reaveraged_count += 1

        if reaveraged_count > 0:
            print(f"  ✓ 共有 {reaveraged_count} 帧经过重新平均处理")
            print(
                f"  ✓ 重新平均比例: {reaveraged_count/len(result_modalities['child_sensor'])*100:.1f}%"
            )
        else:
            print(f"  ⚠️ 没有帧经过重新平均处理")

    except Exception as e:
        print(f"❌ 测试失败，出现异常:")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {str(e)}")
        import traceback

        traceback.print_exc()
