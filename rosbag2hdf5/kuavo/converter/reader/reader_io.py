import glob
import os


class ReaderIOMixin:
    def align_frame_data(self, data: dict):
        aligned_data = defaultdict(list)
        main_timeline = max(
            self.DEFAULT_CAMERA_NAMES, key=lambda cam_k: len(data.get(cam_k, []))
        )
        jump = self.MAIN_TIMELINE_FPS // self.TRAIN_HZ
        main_img_timestamps = [t["timestamp"] for t in data[main_timeline]][
            self.SAMPLE_DROP : -self.SAMPLE_DROP
        ][::jump]
        min_end = min(
            [data[k][-1]["timestamp"] for k in data.keys() if len(data[k]) > 0]
        )
        main_img_timestamps = [t for t in main_img_timestamps if t < min_end]
        for stamp in main_img_timestamps:
            stamp_sec = stamp
            for key, v in data.items():
                if len(v) > 0:
                    this_obs_time_seq = [this_frame["timestamp"] for this_frame in v]
                    time_array = np.array([t for t in this_obs_time_seq])
                    idx = np.argmin(np.abs(time_array - stamp_sec))
                    aligned_data[key].append(v[idx])
                else:
                    aligned_data[key] = []
        log_print(
            f"Aligned {key}: {len((data[main_timeline]))} -> {len(next(iter(aligned_data.values())))}"
        )
        for k, v in aligned_data.items():
            if len(v) > 0:
                log_print(v[0]["timestamp"], v[1]["timestamp"], k)
        return aligned_data

    def list_bag_files(self, bag_dir: str):
        return sorted(glob.glob(os.path.join(bag_dir, "*.bag")))

    def process_rosbag_dir(self, bag_dir: str):
        all_data = []
        # 按照文件名排序，获取 bag 文件列表
        bag_files = self.list_bag_files(bag_dir)
        episode_id = 0
        for bf in bag_files:
            log_print(f"Processing bag file: {bf}")
            episode_data = self.process_rosbag(bf)
            all_data.append(episode_data)

        return all_data


