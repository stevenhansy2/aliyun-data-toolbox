#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Lejurobot 2025
# This file is used to transform the kuavo robot pose to the camera pose.

import numpy as np 
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import BodyIndex
import rosbag
def extract_camera_poses_from_bag_with_time(bag_path, urdf_path, abs_start=None, abs_end=None):
    pose_calculator = KuavoPoseCalculator(urdf_path)
    camera_poses = {
        'timestamps': [],
        'head_camera_poses': [],
        'left_hand_camera_poses': [],
        'right_hand_camera_poses': []
    }
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/sensors_data_raw']):
            t_sec = t.to_sec()
            if abs_start is not None and t_sec < abs_start:
                continue
            if abs_end is not None and t_sec > abs_end:
                continue
            # ...后续逻辑同原函数...
            joint_q = msg.joint_data.joint_q
            left_arm_q = joint_q[12:19]
            right_arm_q = joint_q[19:26]
            head_q = joint_q[-2:]
            try:
                head_camera_pose = pose_calculator.get_camera_pose(head_q)
                left_hand_camera_pose = pose_calculator.get_l_hand_camera_pose(left_arm_q)
                right_hand_camera_pose = pose_calculator.get_r_hand_camera_pose(right_arm_q)
                camera_poses['timestamps'].append(t_sec)
                camera_poses['head_camera_poses'].append(head_camera_pose)
                camera_poses['left_hand_camera_poses'].append(left_hand_camera_pose)
                camera_poses['right_hand_camera_poses'].append(right_hand_camera_pose)
            except ValueError as e:
                print(f"跳过时间戳 {t_sec} 的数据，原因: {e}")
                continue
    return camera_poses
def extract_camera_poses_from_bag(bag_path, urdf_path):
    """
    从rosbag中提取相机位姿数据
    
    Args:
        bag_path (str): rosbag文件路径
        urdf_path (str): URDF文件路径
    
    Returns:
        dict: 包含时间戳和各相机位姿的字典
    """
    # 初始化位姿计算器
    pose_calculator = KuavoPoseCalculator(urdf_path)
    
    # 存储结果
    camera_poses = {
        'timestamps': [],
        'head_camera_poses': [],
        'left_hand_camera_poses': [],
        'right_hand_camera_poses': []
    }
    
    # 打开rosbag
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/sensors_data_raw']):
            # 提取关节角度数据
            joint_q = msg.joint_data.joint_q
            # 提取各部分关节角度
            left_arm_q = joint_q[12:19]  # 左手关节 (索引12-18)
            right_arm_q = joint_q[19:26]  # 右手关节 (索引19-25)
            head_q = joint_q[-2:]  # 头部关节 (最后两个)
            
            try:
                # 计算各相机位姿
                head_camera_pose = pose_calculator.get_camera_pose(head_q)
                left_hand_camera_pose = pose_calculator.get_l_hand_camera_pose(left_arm_q)
                right_hand_camera_pose = pose_calculator.get_r_hand_camera_pose(right_arm_q)
                
                # Debug
                # print("joint_q:", joint_q)
                # print("left_arm_q:", left_arm_q)
                # print("right_arm_q:", right_arm_q)
                # print("head_q:", head_q)
                # print("head_camera_pose:", head_camera_pose)
                # print("left_hand_camera_pose:", left_hand_camera_pose)
                # print("right_hand_camera_pose:", right_hand_camera_pose)
                # return 

                # 存储结果
                camera_poses['timestamps'].append(t.to_sec())
                camera_poses['head_camera_poses'].append(head_camera_pose)
                camera_poses['left_hand_camera_poses'].append(left_hand_camera_pose)
                camera_poses['right_hand_camera_poses'].append(right_hand_camera_pose)
                
            except ValueError as e:
                print(f"跳过时间戳 {t.to_sec()} 的数据，原因: {e}")
                continue
    
    return camera_poses

class KuavoPoseCalculator:
    def __init__(self, urdf_path):
        self.plant = MultibodyPlant(0.0)
        Parser(self.plant).AddModelFromFile(urdf_path)

        self.base_link_frame = self.plant.GetFrameByName("base_link") # 基座坐标系
        self.plant.WeldFrames(self.plant.world_frame(), self.base_link_frame)
        self.plant.Finalize()
        self.context = self.plant.CreateDefaultContext()
        self.nq = self.plant.num_positions() # 关节数
        
        # Debug
        print("----------------------------------------------------------------------------")
        joint_names = self.plant.GetPositionNames(add_model_instance_prefix=False, always_add_suffix=False)
        l_leg_joints = [name for name in joint_names if name.startswith('leg_l')]
        r_leg_joints = [name for name in joint_names if name.startswith('leg_r')]
        l_arm_joints = [name for name in joint_names if name.startswith('zarm_l')]
        r_arm_joints = [name for name in joint_names if name.startswith('zarm_r')]
        head_joints = [name for name in joint_names if name.startswith('zhead')]
        print("关节数量:", self.nq)
        print("左腿关节:", l_leg_joints)
        print("右腿关节:", r_leg_joints)
        print("左臂关节:", l_arm_joints)
        print("右臂关节:", r_arm_joints)
        print("头部关节:", head_joints)
        # for body_index in range(self.plant.num_bodies()):
        #     body = self.plant.get_body(BodyIndex(body_index))
        #     link_name = body.name()
        #     print(f"Link {body_index}: {link_name}")
        print("----------------------------------------------------------------------------")

    def get_camera_pose(self, head_q:list):
        """计算头部相机在基座坐标系中的位姿。
        
        Args:
            head_q (list): 头部关节角度列表, 长度为2, 分别对应头部的两个关节角度(yaw, pitch), 单位为弧度.
            
        Returns:
            RigidTransform: 头部相机在基座坐标系中的位姿变换矩阵。
            
        Raises:
            ValueError: 当head_q长度不为2 或关节角度列表中包含非数字值时抛出异常。
        """
        if len(head_q) != 2:
            raise ValueError(f"head_q must have length 2, but got {len(head_q)}")
        # 检查关节角度列表中是否包含非数字值
        for i, angle in enumerate(head_q):
            if not isinstance(angle, (int, float)) or np.isnan(angle) or np.isinf(angle):
                raise ValueError(f"head_q[{i}] must be a valid number, but got {angle}")
        
        q = np.zeros(self.nq)
        q[-2:] = head_q
        self.plant.SetPositions(self.context, q)
        # fk
        camera_pose_in_base = self.plant.GetFrameByName("camera_base").CalcPose(
            self.context, self.base_link_frame
        )
        return camera_pose_in_base
    
    def get_l_hand_camera_pose(self, larm_q:list):
        """
        计算左臂相机在基座坐标系中的位姿。
        
        Args:
            larm_q (list): 左臂关节角度列表, 长度为7, 分别对应左臂的7个关节角度, 单位为弧度.
            
        Returns:
            RigidTransform: 左臂相机在基座坐标系中的位姿变换矩阵。

        Raises:
            ValueError: 当larm_q长度不为7 或关节角度列表中包含非数字值时抛出异常。
        """
        if len(larm_q) != 7:
            raise ValueError(f"larm_q must have length 7, but got {len(larm_q)}")
        
        # 检查关节角度列表中是否包含非数字值
        for i, angle in enumerate(larm_q):
            if not isinstance(angle, (int, float)) or np.isnan(angle) or np.isinf(angle):
                raise ValueError(f"larm_q[{i}] must be a valid number, but got {angle}")
        
        q = np.zeros(self.nq)
        q[12:19] = larm_q # 左臂关节索引为 12-18 下标从 0 开始
        self.plant.SetPositions(self.context, q)
        # fk
        camera_pose_in_base = self.plant.GetFrameByName("l_hand_camera").CalcPose(
            self.context, self.base_link_frame
        )
        return camera_pose_in_base
    
    def get_r_hand_camera_pose(self, rarm_q:list):
        """
        计算右臂相机在基座坐标系中的位姿。
        
        Args:
            rarm_q (list): 右臂关节角度列表, 长度为7, 分别对应右臂的7个关节角度, 单位为弧度.
            
        Returns:
            RigidTransform: 右臂相机在基座坐标系中的位姿变换矩阵。

        Raises:
            ValueError: 当rarm_q长度不为7 或关节角度列表中包含非数字值时抛出异常。
        """
        if len(rarm_q) != 7:
            raise ValueError(f"rarm_q must have length 7, but got {len(rarm_q)}")
        
        # 检查关节角度列表中是否包含非数字值
        for i, angle in enumerate(rarm_q):
            if not isinstance(angle, (int, float)) or np.isnan(angle) or np.isinf(angle):
                raise ValueError(f"rarm_q[{i}] must be a valid number, but got {angle}")
        
        q = np.zeros(self.nq)
        q[19:26] = rarm_q # 右臂关节索引为 19-25 下标从 0 开始
        self.plant.SetPositions(self.context, q)
        # fk
        camera_pose_in_base = self.plant.GetFrameByName("r_hand_camera").CalcPose(
            self.context, self.base_link_frame
        )
        return camera_pose_in_base

    def get_l_hand_tripod_to_camera_pose(self):
        """
        获取左手三脚架到左手相机的位姿变换。
        
        Returns:
            RigidTransform: 左手三脚架到左手相机的位姿变换矩阵。
        """
        # 获取 l_hand_tripod 和 l_hand_camera 的 frame
        tripod_frame = self.plant.GetFrameByName("l_hand_tripod")
        camera_frame = self.plant.GetFrameByName("l_hand_camera")
        
        # 计算从 l_hand_tripod 到 l_hand_camera 的位姿变换
        tripod_to_camera_pose = camera_frame.CalcPose(self.context, tripod_frame)
        
        return tripod_to_camera_pose
    
    def get_r_hand_tripod_to_camera_pose(self):
        """
        获取右手三脚架到右手相机的位姿变换。
        
        Returns:
            RigidTransform: 右手三脚架到右手相机的位姿变换矩阵。
        """
        # 获取 r_hand_tripod 和 r_hand_camera 的 frame
        tripod_frame = self.plant.GetFrameByName("r_hand_tripod")
        camera_frame = self.plant.GetFrameByName("r_hand_camera")
        
        # 计算从 r_hand_tripod 到 r_hand_camera 的位姿变换
        tripod_to_camera_pose = camera_frame.CalcPose(self.context, tripod_frame)
        
        return tripod_to_camera_pose
    
    def get_head_link_to_camera_pose(self):
        """
        获取头部链接(zhead_2_link)到相机frame(camera)的转换位姿。
        
        Returns:
            RigidTransform: 从zhead_2_link到camera的位姿变换矩阵。
        """
        # 获取zhead_2_link frame
        zhead_2_link_frame = self.plant.GetFrameByName("zhead_2_link")
        # 获取camera frame
        camera_frame = self.plant.GetFrameByName("camera")
        
        # 计算从zhead_2_link到camera的转换
        transform_zhead_to_camera = camera_frame.CalcPose(
            self.context, zhead_2_link_frame
        )
        return transform_zhead_to_camera
    
if __name__ == "__main__":
    # 使用示例 需要替换成您实际的 URDF 路径
    import os
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # urdf_path = os.path.join(script_dir, "biped_s45.urdf")
    # print("URDF path:", urdf_path)
    urdf_path = "./kuavo/assets/urdf/biped_s45.urdf"
    kuavo_camera_pose_calculator = KuavoPoseCalculator(urdf_path)
    
    # head_camera_pose = kuavo_camera_pose_calculator.get_camera_pose([0.0, 0.0])
    # print("head_camera pose:", head_camera_pose)
    
    # l_hand_camera_pose = kuavo_camera_pose_calculator.get_l_hand_camera_pose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # print("lhand_camera pose:", l_hand_camera_pose)
    
    pose = kuavo_camera_pose_calculator.get_l_hand_tripod_to_camera_pose()
    print("l_hand_tripod_to_camera_pose:", pose)
    
    # r_hand_camera_pose = kuavo_camera_pose_calculator.get_r_hand_camera_pose([0.25, -1.0, -1.0, -1.208, 0.156, 0.658, 0.13])
    # print("rhand_camera pose: ", r_hand_camera_pose)

    pose1 = kuavo_camera_pose_calculator.get_r_hand_tripod_to_camera_pose()
    print("r_hand_tripod_to_camera_pose:", pose1)

    pose2 = kuavo_camera_pose_calculator.get_head_link_to_camera_pose()
    print("head_link_to_camera_pose:", pose2)