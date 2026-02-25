import os

import numpy as np


class IncrementalHDF5Writer:
    """
    增量 HDF5 写入器：支持分批追加数据到可扩展 dataset。

    用法:
        writer = IncrementalHDF5Writer("output.hdf5")
        writer.create_dataset("data/joint_q", shape=(0, 14), maxshape=(None, 14), dtype=np.float32)
        for batch in batches:
            writer.append("data/joint_q", batch)  # batch shape: (N, 14)
        writer.close()
    """

    def __init__(self, file_path: str, batch_size: int = 5000):
        import h5py
        self.file_path = file_path
        self.batch_size = batch_size
        self._file = None
        self._datasets = {}
        self._row_counts = {}
        self._append_counts = {}

        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
        self._file = h5py.File(file_path, "w")

    def create_dataset(
        self,
        name: str,
        shape: tuple = None,
        maxshape: tuple = None,
        dtype=np.float32,
        chunks: tuple = None,
    ):
        """
        创建可扩展 dataset。

        Args:
            name: dataset 路径 (如 "data/joint_q")
            shape: 初始形状，第一维通常为 0
            maxshape: 最大形状，第一维为 None 表示可扩展
            dtype: 数据类型
            chunks: chunk 大小，默认自动计算
        """
        if shape is None:
            shape = (0,)
        if maxshape is None:
            maxshape = (None,) + shape[1:]
        if chunks is None and len(shape) > 0:
            chunks = (min(self.batch_size, 1000),) + shape[1:]

        parts = name.split("/")
        group = self._file
        for part in parts[:-1]:
            if part not in group:
                group = group.create_group(part)
            else:
                group = group[part]

        ds = group.create_dataset(
            parts[-1],
            shape=shape,
            maxshape=maxshape,
            dtype=dtype,
            chunks=chunks,
        )
        self._datasets[name] = ds
        self._row_counts[name] = 0
        self._append_counts[name] = 0
        return ds

    def append(self, name: str, data: np.ndarray):
        """
        追加数据到 dataset。

        Args:
            name: dataset 路径
            data: 要追加的数据 (numpy array)
        """
        if name not in self._datasets:
            raise KeyError(f"Dataset {name} not found. Call create_dataset first.")

        ds = self._datasets[name]
        data = np.asarray(data)

        if data.ndim == 1 and len(ds.shape) > 1:
            data = data.reshape(1, -1)

        current_rows = ds.shape[0]
        new_rows = data.shape[0]

        ds.resize(current_rows + new_rows, axis=0)
        ds[current_rows:current_rows + new_rows] = data

        self._row_counts[name] += new_rows
        self._append_counts[name] += 1

    def get_stats(self) -> dict:
        """获取写入统计信息"""
        return {
            name: {
                "rows": self._row_counts[name],
                "appends": self._append_counts[name],
            }
            for name in self._datasets
        }

    def close(self):
        """关闭 HDF5 文件"""
        if self._file:
            stats = self.get_stats()
            for name, info in stats.items():
                log_print(f"[IncrementalHDF5Writer] {name}: {info['rows']} rows, {info['appends']} appends")
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def write_dict_to_hdf5_batched(
    data_dict: dict,
    file_path: str,
    batch_size: int = 10000,
    large_threshold: int = 10000,
) -> dict:
    """
    将嵌套 dict 写入 HDF5，对大数据集使用分批写入。

    Args:
        data_dict: 嵌套的数据 dict
        file_path: 输出文件路径
        batch_size: 每批写入的行数
        large_threshold: 超过此行数的数据集使用分批写入

    Returns:
        统计信息 dict
    """
    import h5py

    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)

    stats = {"total_datasets": 0, "batched_datasets": 0, "total_rows": 0}
    str_dt = h5py.string_dtype(encoding="utf-8")

    def get_array_and_dtype(value):
        """将值转换为 numpy array 并确定 dtype"""
        if value is None:
            return np.array([], dtype=str_dt), str_dt
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return np.array([], dtype=np.float32), np.float32
            if all(isinstance(x, str) for x in value):
                return np.array(value, dtype=str_dt), str_dt
            return np.array(value), None

        if isinstance(value, np.ndarray):
            if value.dtype == object:
                try:
                    return np.array([str(x) for x in value.flat], dtype=str_dt).reshape(value.shape), str_dt
                except:
                    return value, None
            return value, None

        return np.array([value]), None

    def write_recursively(group, d, path=""):
        for key, value in d.items():
            full_path = f"{path}/{key}" if path else key

            if isinstance(value, dict):
                subgroup = group.create_group(key)
                write_recursively(subgroup, value, full_path)
            else:
                arr, dtype = get_array_and_dtype(value)
                n_rows = arr.shape[0] if arr.ndim > 0 else 1

                stats["total_datasets"] += 1
                stats["total_rows"] += n_rows

                if n_rows > large_threshold:
                    stats["batched_datasets"] += 1
                    maxshape = (None,) + arr.shape[1:]
                    chunks = (min(batch_size, n_rows),) + arr.shape[1:]

                    ds = group.create_dataset(
                        key,
                        shape=arr.shape,
                        maxshape=maxshape,
                        dtype=dtype if dtype else arr.dtype,
                        chunks=chunks,
                    )

                    for start in range(0, n_rows, batch_size):
                        end = min(start + batch_size, n_rows)
                        ds[start:end] = arr[start:end]

                    log_print(f"[HDF5 Batched] {full_path}: {n_rows} rows in {(n_rows + batch_size - 1) // batch_size} batches")
                else:
                    if dtype:
                        group.create_dataset(key, data=arr, dtype=dtype)
                    else:
                        group.create_dataset(key, data=arr)

    with h5py.File(file_path, "w") as f:
        write_recursively(f, data_dict)

    log_print(f"[HDF5] 写入完成: {stats['total_datasets']} datasets, {stats['batched_datasets']} batched, {stats['total_rows']} total rows")
    return stats
