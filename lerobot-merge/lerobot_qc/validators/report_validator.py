from typing import Optional, Dict
from utils.report_loader import ReportLoader

class ReportValidator:
    def __init__(self, report:ReportLoader, will_fix_data = False, episodes_length_dict = None):
        self.report = report
        self.episodes_length_dict = episodes_length_dict
        self.will_fix_data = will_fix_data
        self.pending = []
        self.discard = []


    def classify_by_count(self, type_, key, threshold):
        """
        type_: 'gripper' or 'angle'
        key: gripper的key或者angle的key
        threshold: 判断阈值，out_of_range_count超出多少个视为discard
        """
        status_dict = {"success": [], "pending": [], "discard": []}
        for idx, episode in enumerate(self.report.episode_validations):
            count = None
            pending_count = 0
            try:
                if type_ == "gripper":
                    gripper = self.report.get_gripper(idx)
                    pending_count = gripper[key]["tolerance_range_count"]
                    count = gripper[key]["out_of_range_count"]
                else:
                    count = self.report.get_angle(idx)[key]["out_of_range_count"]
            except Exception:
                count = None
            if isinstance(count, int):
                if count == 0 and pending_count == 0:
                    status_dict["success"].append(idx)
                elif count < threshold:
                    status_dict["pending"].append(idx)
                else:
                    status_dict["discard"].append(idx)
            else:
                status_dict["discard"].append(idx)
        return status_dict

        """
        type_: 'gripper' or 'angle'
        key: gripper的key或者angle的key
        没有阈值，出
        """
        status_dict = {"success": [], "pending": [], "discard": []}
        for idx, episode in enumerate(self.report.episode_validations):
            count = None
            pending_count = 0
            try:
                if type_ == "gripper":
                    gripper = self.report.get_gripper(idx)
                    pending_count = gripper[key]["tolerance_range_count"]
                    count = gripper[key]["out_of_range_count"]
                else:
                    count = self.report.get_angle(idx)[key]["out_of_range_count"]
            except Exception:
                count = None
            if isinstance(count, int):
                if count == 0 and pending_count == 0:
                    status_dict["success"].append(idx)
                elif count < threshold:
                    status_dict["pending"].append(idx)
                else:
                    status_dict["discard"].append(idx)
            else:
                status_dict["discard"].append(idx)
        return status_dict

    def summarize_count_status(self, threshold_gripper=0, threshold_angle=0):
        """
        返回所有gripper key和angle key的状态分类
        """
        output = {"gripper": {}, "angle": {}}
        gripper_keys = self.report.collect_gripper_keys()
        angle_keys = self.report.collect_angle_keys()
        # 1. gripper
        for key in gripper_keys:
            output["gripper"][key] = self.classify_by_count("gripper", key, threshold_gripper)
        # 2. angle
        for key in angle_keys:
            output["angle"][key] = self.classify_by_count("angle", key, threshold_angle)
        return output

    def out_of_range_indices_including_ends(self, idx):
        gripper_keys = self.report.collect_gripper_keys()
        angle_keys = self.report.collect_angle_keys()
        angle = self.report.get_angle(idx)
        frame_lenth = self.episodes_length_dict.get(idx, -1)
        end_frame_idx = frame_lenth - 1
        for key in angle_keys:
            if angle[key]['out_of_range_count'] > 0:
                idxs = angle[key]['out_of_range_indices']
                if 0 in idxs or end_frame_idx in idxs:
                    return True
        gripper = self.report.get_gripper(idx)
        for key in gripper_keys:
            if gripper[key]['out_of_range_count'] > 0:
                idxs = gripper[key]['out_of_range_indices']
                if 0 in idxs or end_frame_idx in idxs:
                    return True
        return False


    def episode_final_status(self, threshold_gripper=0, threshold_angle=0):
        """
        综合所有key，优先级：废弃>待定>成功
        返回每个episode_idx最终状态 {idx: status}
        只要有任何key被判为discard就为discard，否则只要有pending就是pending，否则是success
        """
        if not self.report.get_level0_validation_result() or not self.report.get_dataset_validation_result():
            # 最低限度的检测就不通过
            return None

        n = len(self.report.episode_validations)
        # self.pending = []
        self.discard = []
        for idx, val in enumerate(self.report.episode_validations):
            v = val.get('passed', False)
            if not v:
                self.discard.append(val.get('episode_idx', idx))
        
        return True
