import os

import h5py
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
            log_print(
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
            log_print(
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
            log_print(
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
        """将数据保存为符合库帕思通用版数据格式的HDF5文件（支持字符串列表写入）"""
        import h5py

        # 确保输出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 可变长 UTF-8 字符串 dtype
        str_dt = h5py.string_dtype(encoding="utf-8")

        def process_data_by_path(value, path):
            """根据数据路径对数据进行特殊处理并返回 (data, dtype_override_or_None)"""
            # None 直接返回
            if value is None:
                return np.array([], dtype=str_dt), str_dt

            # 如果是 list/tuple，先转为 numpy 数组（可能是字符串列表）
            if isinstance(value, (list, tuple)):
                # 如果元素是 bytes，保持 bytes；否则转为 str
                if all(isinstance(x, (bytes, bytearray)) for x in value):
                    arr = np.array(value, dtype=str_dt)
                    return arr, None
                # 检查是否全为数字
                if all((isinstance(x, (int, float, np.integer, np.floating)) or (hasattr(x, "dtype") and np.issubdtype(getattr(x, "dtype"), np.number))) for x in value):
                    arr = np.array(value)
                    return arr, None
                # 其它情况转为字符串数组（统一 utf-8）
                str_list = ["" if x is None else str(x) for x in value]
                arr = np.array(str_list, dtype=str_dt)
                return arr, str_dt

            # 如果是 numpy 数组或标量
            if hasattr(value, "dtype"):
                # 处理固定长度 bytes('S') 或 numpy bytes_ dtype（例如 b'leg_l1_joint'）
                if np.issubdtype(value.dtype, np.bytes_) or getattr(value.dtype, "kind", "") == "S":
                    decoded = [
                        (x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x))
                        for x in np.ravel(value)
                    ]
                    arr = np.array(decoded, dtype=str_dt).reshape(value.shape)
                    return arr, str_dt

                # 处理 object dtype，可能包含 bytes 或 str 或 None
                if value.dtype == object:
                    flat = list(np.ravel(value))
                    if all(isinstance(x, (bytes, bytearray)) for x in flat):
                        decoded = [x.decode("utf-8") for x in flat]
                        arr = np.array(decoded, dtype=str_dt).reshape(value.shape)
                        return arr, str_dt
                    if all((x is None) or isinstance(x, str) for x in flat):
                        normalized = ["" if x is None else x for x in flat]
                        arr = np.array(normalized, dtype=str_dt).reshape(value.shape)
                        return arr, str_dt

                # unicode / numpy.str_ 普通处理为 utf-8 可变长字符串
                if np.issubdtype(value.dtype, np.str_) or np.issubdtype(value.dtype, np.unicode_):
                    arr = np.array([str(x) for x in value.reshape(-1)], dtype=str_dt).reshape(value.shape)
                    return arr, str_dt

                # 数值类型直接返回
                if np.issubdtype(value.dtype, np.number):
                    return value, None
                # 其它回退为字符串数组
                flat = ["" if x is None else str(x) for x in np.ravel(value)]
                arr = np.array(flat, dtype=str_dt).reshape(value.shape)
                return arr, str_dt

            # 其他回退为字符串
            return np.array([str(value)], dtype=str_dt), str_dt

        def create_datasets_recursively(group, data_dict, current_path=""):
            """递归创建数据集和组，自动处理字符串列表类型"""
            for key, value in data_dict.items():
                full_path = f"{current_path}/{key}" if current_path else key

                if isinstance(value, dict):
                    subgroup = group.create_group(key)
                    create_datasets_recursively(subgroup, value, full_path)
                else:
                    try:
                        data_processed, dtype_hint = process_data_by_path(value, full_path)

                        # 如果 dtype_hint 是 str_dt，则确保以 utf-8 可变长字符串写入
                        if dtype_hint is str_dt:
                            # flatten if necessary and preserve shape
                            try:
                                arr = np.array(["" if x is None else str(x) for x in data_processed.reshape(-1)], dtype=str_dt)
                                arr = arr.reshape(data_processed.shape)
                                group.create_dataset(key, data=arr, dtype=str_dt)
                            except Exception:
                                # fallback: create 1D string array
                                arr = np.array(["" if x is None else str(x) for x in np.ravel(data_processed)], dtype=str_dt)
                                group.create_dataset(key, data=arr, dtype=str_dt)
                        else:
                            # 如果 data_processed 是 object dtype with strings -> convert to str_dt
                            if isinstance(data_processed, np.ndarray) and data_processed.dtype == object:
                                try:
                                    arr = np.array(["" if x is None else str(x) for x in data_processed.reshape(-1)], dtype=str_dt)
                                    arr = arr.reshape(data_processed.shape)
                                    group.create_dataset(key, data=arr, dtype=str_dt)
                                    dtype_hint = str_dt
                                except Exception:
                                    arr = np.array(["" if x is None else str(x) for x in np.ravel(data_processed)], dtype=str_dt)
                                    group.create_dataset(key, data=arr, dtype=str_dt)
                                    dtype_hint = str_dt
                            else:
                                # 数值或 bytes 等，直接写入
                                group.create_dataset(key, data=data_processed)
                                dtype_hint = data_processed.dtype
                        log_print(f"创建数据集: {full_path}, 形状: {np.array(value).shape if value is not None else (0,)}, 类型提示: {dtype_hint}")
                    except Exception as e:
                        log_print(f"警告: 无法创建数据集 {full_path}: {e}")
                        # 创建空数据集占位（字符串类型）
                        try:
                            group.create_dataset(key, data=np.array([], dtype=str_dt), dtype=str_dt)
                        except Exception:
                            pass

        def add_missing_required_fields(f, low_dim_data):
            """添加库帕思格式中必需但缺失的字段，使用null机制"""
            # 保留原实现（调用原来的 inner 函数块）
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
                if dtype == np.float32 or dtype == np.float64:
                    data = np.full(shape, np.nan, dtype=dtype)
                    dataset = group.create_dataset(name, data=data)
                    dataset.attrs["missing_data"] = True
                    dataset.attrs["description"] = f"Missing data filled with NaN"
                    return dataset

                # 方法2: 创建空数据集（对于整数类型）
                elif np.issubdtype(dtype, np.integer):
                    if dtype == np.int32:
                        fill_value = np.iinfo(np.int32).min
                    elif dtype == np.int64:
                        fill_value = np.iinfo(np.int64).min
                    else:
                        fill_value = -999999
                    data = np.full(shape, fill_value, dtype=dtype)
                    dataset = group.create_dataset(name, data=data)
                    dataset.attrs["missing_data"] = True
                    dataset.attrs["fill_value"] = fill_value
                    dataset.attrs["description"] = f"Missing data filled with {fill_value}"
                    return dataset

                else:
                    missing_group = group.create_group(name + "_missing")
                    missing_group.attrs["missing_data"] = True
                    missing_group.attrs["expected_shape"] = shape
                    missing_group.attrs["expected_dtype"] = str(dtype)
                    missing_group.attrs["description"] = "Data not available - missing field"
                    return missing_group

            def create_optional_dataset(group, name, shape, dtype, description="Optional field not available"):
                empty_data = np.array([], dtype=dtype)
                dataset = group.create_dataset(name, data=empty_data, maxshape=shape)
                dataset.attrs["data_available"] = False
                dataset.attrs["expected_shape"] = shape
                dataset.attrs["description"] = description
                return dataset

            # 以下保留原有逻辑（摘录并使用现有实现）
            # 检查并添加缺失的 action 组字段
            if "action" in f:
                action_group = f["action"]
                # 保持注释块不变（略）

            # 检查并添加缺失的 state 组字段
            if "state" in f:
                state_group = f["state"]
                if "robot" not in state_group:
                    robot_group = state_group.create_group("robot")
                    if "imu" in low_dim_data and "quat_xyzw" in low_dim_data["imu"]:
                        imu_data_quat_xyzw = low_dim_data["imu"]["quat_xyzw"]
                        if (
                            hasattr(imu_data_quat_xyzw, "shape")
                            and len(imu_data_quat_xyzw.shape) > 1
                            and imu_data_quat_xyzw.shape[1] >= 4
                        ):
                            # 简单下采样/填充以防长度不一致（保持之前的兼容策略）
                            imu_len = imu_data_quat_xyzw.shape[0]
                            if imu_len == N:
                                orientation = imu_data_quat_xyzw.astype(np.float32)
                            elif imu_len > N:
                                indices = np.linspace(0, imu_len - 1, N).astype(int)
                                orientation = imu_data_quat_xyzw[indices, :].astype(np.float32)
                            else:
                                orientation = np.zeros((N, 4), dtype=np.float32)
                                minlen = imu_len
                                orientation[:minlen, :] = imu_data_quat_xyzw.astype(np.float32)
                                orientation[minlen:, :] = imu_data_quat_xyzw[minlen - 1, :].astype(np.float32)

                            dataset = robot_group.create_dataset("orientation", data=orientation)
                            dataset.attrs["data_source"] = "IMU sensor"
                            dataset.attrs["missing_data"] = False
                            log_print(f"从IMU数据提取机器人姿态")
                        else:
                            create_null_dataset(robot_group, "orientation", (N, 4), np.float32)
                            log_print(f"IMU数据格式异常，姿态数据标记为缺失")
                    else:
                        create_null_dataset(robot_group, "orientation", (N, 4), np.float32)
                        log_print(f"无IMU数据，姿态数据标记为缺失")

                    log_print(f"添加缺失字段: state/robot (使用NaN/缺失值表示)")

                if "head" in state_group:
                    head_group = state_group["head"]
                    if "effort" not in head_group:
                        create_null_dataset(head_group, "effort", (N, 2), np.float32)
                        log_print(f"添加缺失字段: state/head/effort (使用NaN表示缺失)")

                if "joint" in state_group:
                    joint_group = state_group["joint"]
                    joint_count = 14
                    if "position" in joint_group:
                        joint_count = joint_group["position"].shape[1]
                    elif "velocity" in joint_group:
                        joint_count = joint_group["velocity"].shape[1]

                    if "current_value" not in joint_group:
                        create_null_dataset(joint_group, "current_value", (N, joint_count), np.float32)
                        log_print(f"添加缺失字段: state/joint/current_value (使用NaN表示缺失)")

                    if "effort" not in joint_group:
                        create_null_dataset(joint_group, "effort", (N, joint_count), np.float32)
                        log_print(f"添加缺失字段: state/joint/effort (使用NaN表示缺失)")

        # 写入 HDF5 文件
        with h5py.File(file_path, "w") as f:
            log_print(f"开始创建HDF5文件: {file_path}")
            create_datasets_recursively(f, low_dim_data)
            add_missing_required_fields(f, low_dim_data)

        log_print(f"数据已成功保存为HDF5格式: {file_path}")
        return file_path
    # def save_to_hdf5(low_dim_data, file_path):
    #     """将数据保存为符合库帕思通用版数据格式的HDF5文件"""
    #     import h5py

    #     # 确保输出目录存在
    #     os.makedirs(os.path.dirname(file_path), exist_ok=True)

    #     def create_datasets_recursively(group, data_dict, current_path=""):
    #         """递归创建数据集和组"""
    #         for key, value in data_dict.items():
    #             full_path = f"{current_path}/{key}" if current_path else key

    #             if isinstance(value, dict):
    #                 # 如果是字典，创建子组并递归处理
    #                 subgroup = group.create_group(key)
    #                 create_datasets_recursively(subgroup, value, full_path)
    #             else:
    #                 # 如果是数据，创建数据集
    #                 try:
    #                     # 处理不同类型的数据
    #                     if isinstance(value, (list, tuple)):
    #                         value = np.array(value)

    #                     # 根据数据类型和路径进行特殊处理
    #                     processed_value = process_data_by_path(value, full_path)

    #                     # 创建数据集
    #                     group.create_dataset(key, data=processed_value)
    #                     log_print(
    #                         f"创建数据集: {full_path}, 形状: {processed_value.shape}, 类型: {processed_value.dtype}"
    #                     )

    #                 except Exception as e:
    #                     log_print(f"警告: 无法创建数据集 {full_path}: {e}")
    #                     # 创建空数据集作为占位符
    #                     try:
    #                         empty_data = np.array([])
    #                         group.create_dataset(key, data=empty_data)
    #                     except:
    #                         pass

    #     def process_data_by_path(value, path):
    #         """根据数据路径对数据进行特殊处理"""
    #         # 时间戳处理 - 扩展识别新的时间戳字段
    #         timestamp_fields = [
    #             "timestamps",
    #             "head_color_mp4_camera_timestamps",
    #             "hand_left_color_mp4_timestamps",
    #             "hand_right_color_mp4_timestamps",
    #             "head_depth_mkv_camera_timestamps",
    #             "hand_left_depth_mkv_timestamps",
    #             "hand_right_depth_mkv_timestamps",
    #             "camera_extrinsics_timestamps",
    #             "head_timestamps" "joint_timestamps",
    #             "effector_dexhand_timestamps",
    #             "effector_lejuclaw_timestamps",
    #         ]

    #         if any(ts_field in path for ts_field in timestamp_fields):
    #             if value.dtype != np.int64:
    #                 # 转换时间戳为纳秒级整数
    #                 if np.issubdtype(value.dtype, np.floating):
    #                     return (value * 1e9).astype(np.int64)
    #                 else:
    #                     return value.astype(np.int64)
    #             return value

    #         # 索引数据处理
    #         elif "index" in path:
    #             return value.astype(np.int64)

    #         # 其他数值数据处理
    #         elif np.issubdtype(value.dtype, np.number):
    #             # 根据数据类型决定精度
    #             if np.issubdtype(value.dtype, np.integer):
    #                 return value.astype(np.int32)
    #             else:
    #                 return value.astype(np.float32)

    #         # 保持原始数据类型
    #         return value

    #     def add_missing_required_fields(f, low_dim_data):
    #         """添加库帕思格式中必需但缺失的字段，使用null机制"""

    #         # 获取时间戳长度作为参考
    #         if "timestamps" in low_dim_data:
    #             N = len(low_dim_data["timestamps"])
    #         else:
    #             N = 1000  # 默认值
    #             for key, value in low_dim_data.items():
    #                 if hasattr(value, "__len__") and not isinstance(value, str):
    #                     N = len(value)
    #                     break

    #         # 创建控制索引
    #         control_indices = np.arange(N, dtype=np.int64)

    #         def create_null_dataset(group, name, shape, dtype):
    #             """创建一个表示缺失数据的数据集"""
    #             # 方法1: 使用NaN表示缺失数据（仅适用于浮点数）
    #             if dtype == np.float32 or dtype == np.float64:
    #                 data = np.full(shape, np.nan, dtype=dtype)
    #                 dataset = group.create_dataset(name, data=data)
    #                 # 添加属性标记这是缺失数据
    #                 dataset.attrs["missing_data"] = True
    #                 dataset.attrs["description"] = f"Missing data filled with NaN"
    #                 return dataset

    #             # 方法2: 创建空数据集（对于整数类型）
    #             elif np.issubdtype(dtype, np.integer):
    #                 # 对于整数，使用最小值表示缺失
    #                 if dtype == np.int32:
    #                     fill_value = np.iinfo(np.int32).min
    #                 elif dtype == np.int64:
    #                     fill_value = np.iinfo(np.int64).min
    #                 else:
    #                     fill_value = -999999  # 默认缺失值

    #                 data = np.full(shape, fill_value, dtype=dtype)
    #                 dataset = group.create_dataset(name, data=data)
    #                 dataset.attrs["missing_data"] = True
    #                 dataset.attrs["fill_value"] = fill_value
    #                 dataset.attrs["description"] = (
    #                     f"Missing data filled with {fill_value}"
    #                 )
    #                 return dataset

    #             # 方法3: 不创建数据集，仅添加占位符属性
    #             else:
    #                 # 创建一个只有属性的组来表示缺失
    #                 missing_group = group.create_group(name + "_missing")
    #                 missing_group.attrs["missing_data"] = True
    #                 missing_group.attrs["expected_shape"] = shape
    #                 missing_group.attrs["expected_dtype"] = str(dtype)
    #                 missing_group.attrs["description"] = (
    #                     "Data not available - missing field"
    #                 )
    #                 return missing_group

    #         def create_optional_dataset(
    #             group, name, shape, dtype, description="Optional field not available"
    #         ):
    #             """创建可选的数据集，明确标记为不可用"""
    #             # 方法4: 创建虚拟数据集，长度为0
    #             empty_data = np.array([], dtype=dtype)
    #             dataset = group.create_dataset(name, data=empty_data, maxshape=shape)
    #             dataset.attrs["data_available"] = False
    #             dataset.attrs["expected_shape"] = shape
    #             dataset.attrs["description"] = description
    #             return dataset

    #         # 检查并添加缺失的 action 组字段
    #         if "action" in f:
    #             action_group = f["action"]

    #             # # 添加缺失的 robot 组
    #             # if "robot" not in action_group:
    #             #     robot_group = action_group.create_group("robot")
    #             #     create_null_dataset(robot_group, "velocity", (N, 2), np.float32)
    #             #     create_null_dataset(robot_group, "index", (N,), np.float32)
    #             #     log_print(f"添加缺失字段: action/robot (使用NaN表示缺失)")

    #             # # 添加缺失的 waist 组
    #             # if "waist" not in action_group:
    #             #     waist_group = action_group.create_group("waist")
    #             #     create_null_dataset(waist_group, "position", (N, 2), np.float32)
    #             #     create_null_dataset(waist_group, "index", (N,), np.float32)
    #             #     log_print(f"添加缺失字段: action/waist (使用NaN表示缺失)")

    #             # # 添加缺失的 end 组
    #             # if "end" not in action_group:
    #             #     end_group = action_group.create_group("end")
    #             #     create_null_dataset(end_group, "orientation", (N, 2, 4), np.float32)
    #             #     create_null_dataset(end_group, "position", (N, 2, 3), np.float32)
    #             #     create_null_dataset(end_group, "index", (N,), np.float32)
    #             #     log_print(f"添加缺失字段: action/end (使用NaN表示缺失)")

    #         # 检查并添加缺失的 state 组字段
    #         if "state" in f:
    #             state_group = f["state"]

    #             # # 添加缺失的 end 组
    #             # if "end" not in state_group:
    #             #     end_group = state_group.create_group("end")
    #             #     create_null_dataset(end_group, "angular", (N, 2, 3), np.float32)
    #             #     create_null_dataset(end_group, "orientation", (N, 2, 4), np.float32)
    #             #     create_null_dataset(end_group, "position", (N, 2, 3), np.float32)
    #             #     create_null_dataset(end_group, "velocity", (N, 2, 3), np.float32)
    #             #     create_null_dataset(end_group, "wrench", (N, 2, 6), np.float32)
    #             #     log_print(f"添加缺失字段: state/end (使用NaN表示缺失)")

    #             # 添加缺失的 robot 组
    #             if "robot" not in state_group:
    #                 robot_group = state_group.create_group("robot")

    #                 # 对于机器人姿态，如果没有IMU数据，明确标记为缺失
    #                 if "imu" in low_dim_data and "quat_xyzw" in low_dim_data["imu"]:
    #                     imu_data_quat_xyzw = low_dim_data["imu"]["quat_xyzw"]
    #                     if (
    #                         hasattr(imu_data_quat_xyzw, "shape")
    #                         and len(imu_data_quat_xyzw.shape) > 1
    #                         and imu_data_quat_xyzw.shape[1] >= 4
    #                     ):
    #                         # 有IMU数据，直接使用
    #                         orientation = np.zeros((N, 4), dtype=np.float32)
    #                         orientation[:, :] = imu_data_quat_xyzw
    #                         dataset = robot_group.create_dataset(
    #                             "orientation", data=orientation
    #                         )
    #                         dataset.attrs["data_source"] = "IMU sensor"
    #                         dataset.attrs["missing_data"] = False
    #                         log_print(f"从IMU数据提取机器人姿态")
    #                     else:
    #                         # IMU数据格式不对，标记为缺失
    #                         create_null_dataset(
    #                             robot_group, "orientation", (N, 4), np.float32
    #                         )
    #                         log_print(f"IMU数据格式异常，姿态数据标记为缺失")
    #                 else:
    #                     # 没有IMU数据，标记为缺失
    #                     create_null_dataset(
    #                         robot_group, "orientation", (N, 4), np.float32
    #                     )
    #                     log_print(f"无IMU数据，姿态数据标记为缺失")

    #                 # 其他机器人状态标记为缺失
    #                 # create_null_dataset(robot_group, "orientation_drift", (N, 4), np.float32)
    #                 # create_null_dataset(robot_group, "position", (N, 3), np.float32)
    #                 # create_null_dataset(robot_group, "position_drift", (N, 3), np.float32)
    #                 log_print(f"添加缺失字段: state/robot (使用NaN/缺失值表示)")

    #             # # 添加缺失的 waist 组
    #             # if "waist" not in state_group:
    #             #     waist_group = state_group.create_group("waist")
    #             #     create_null_dataset(waist_group, "effort", (N, 2), np.float32)
    #             #     create_null_dataset(waist_group, "position", (N, 2), np.float32)
    #             #     create_null_dataset(waist_group, "velocity", (N, 2), np.float32)
    #             #     log_print(f"添加缺失字段: state/waist (使用NaN表示缺失)")

    #             # 为现有组添加缺失的数据集
    #             # if "effector" in state_group:
    #             #     effector_group = state_group["effector"]
    #             #     if "force" not in effector_group:
    #             #         create_null_dataset(effector_group, "force", (N, 2), np.float32)
    #             #         log_print(f"添加缺失字段: state/effector/force (使用NaN表示缺失)")

    #             if "head" in state_group:
    #                 head_group = state_group["head"]
    #                 if "effort" not in head_group:
    #                     create_null_dataset(head_group, "effort", (N, 2), np.float32)
    #                     log_print(f"添加缺失字段: state/head/effort (使用NaN表示缺失)")

    #             if "joint" in state_group:
    #                 joint_group = state_group["joint"]
    #                 # 获取关节数量
    #                 joint_count = 14  # 默认值
    #                 if "position" in joint_group:
    #                     joint_count = joint_group["position"].shape[1]
    #                 elif "velocity" in joint_group:
    #                     joint_count = joint_group["velocity"].shape[1]

    #                 if "current_value" not in joint_group:
    #                     create_null_dataset(
    #                         joint_group, "current_value", (N, joint_count), np.float32
    #                     )
    #                     log_print(
    #                         f"添加缺失字段: state/joint/current_value (使用NaN表示缺失)"
    #                     )

    #                 if "effort" not in joint_group:
    #                     create_null_dataset(
    #                         joint_group, "effort", (N, joint_count), np.float32
    #                     )
    #                     log_print(f"添加缺失字段: state/joint/effort (使用NaN表示缺失)")

    #         # 添加 other_sensors 组（标记为可选）
    #         # if "other_sensors" not in f:
    #         #     other_group = f.create_group("other_sensors")
    #         #     other_group.attrs['description'] = 'Optional sensor data - currently empty'
    #         #     other_group.attrs['data_available'] = False
    #         #     log_print(f"添加缺失字段: other_sensors (标记为可选数据)")
    #         # 新增：在根级别添加时间戳字段的存在性信息

    #     # 创建 HDF5 文件
    #     with h5py.File(file_path, "w") as f:
    #         log_print(f"开始创建HDF5文件: {file_path}")

    #         # 递归创建所有数据集和组
    #         create_datasets_recursively(f, low_dim_data)

    #         # 添加库帕思格式要求的缺失字段填充为NaN或缺失值
    #         add_missing_required_fields(f, low_dim_data)

    #     log_print(f"数据已成功保存为HDF5格式: {file_path}")
    #     return file_path
