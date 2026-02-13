import json
import os

class ReportLoader:
    def __init__(self, content):
        self.data = content

    @property
    def task_name(self):
        return self.data.get('task_name', '')

    @property
    def from_oss(self):
        return self.data.get('from_oss', False)

    @property
    def config_path(self):
        return self.data.get('config_path', '')
    
    @property
    def level0_validation(self):
        return self.data.get('level0_validation', {})

    @property
    def dataset_validation(self):
        return self.data.get('dataset_validation', {})

    @property
    def episode_validations(self):
        return self.data.get('episode_validations', [])

    def get_level0_validation_result(self):
        return self.level0_validation.get('passed', False)
    
    def get_dataset_validation_result(self):
        return self.dataset_validation.get('passed', False)

    def get_episode_count(self):
        return len(self.episode_validations)

    def get_episode(self, idx):
        """获取第idx个 episode_validation（从0开始）"""
        if idx < 0 or idx >= self.get_episode_count():
            return None
        return self.episode_validations[idx]
    
    def get_gripper(self, idx):
        """获取第idx个 gripper_check_results（从0开始）"""
        episode = self.get_episode(idx)
        if episode is None:
            return None
        try:
            data = episode["B2_angle_gripper_check"]["gripper_check_results"]
            return data
        except Exception:
            logging.error(f"There is no B2_angle_gripper_check.gripper_check_results for episode {idx}")
            return None 

    def get_angle(self, idx):
        """获取第idx个 angle_domain_results（从0开始）"""
        episode = self.get_episode(idx)
        if episode is None:
            return None
        try:
            data = episode["B2_angle_gripper_check"]["angle_domain_results"]
            return data
        except Exception:
            logging.error(f"There is no B2_angle_gripper_check.angle_domain_results for episode {idx}")
            return None 

    def safe_get_keys(self, episode, *keys_path):
        if episode is None:
            return set()
        current = episode
        for key in keys_path:
            if not isinstance(current, dict) or key not in current:
                return set()
            current = current[key]
        return set(current.keys())
    
    def collect_gripper_keys(self):
        episode = self.get_episode(0)
        return list(self.safe_get_keys(episode, "B2_angle_gripper_check", "gripper_check_results"))

    def collect_angle_keys(self):
        episode = self.get_episode(0)
        return list(self.safe_get_keys(episode, "B2_angle_gripper_check", "angle_domain_results"))