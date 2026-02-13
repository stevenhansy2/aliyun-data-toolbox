# Lerobot数据格式说明

该数据集基于lerobotv21版本，汇总了现有单个bag包中有用的信息。
安装方式：
```
conda create -y -n lerobot python=3.10
conda activate lerobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
pip install joblib -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy==1.26.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install drake==1.19.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
# 运行方式
在刻行平台上运行：
1、打包docker
2、上传至刻行指定项目
3、在刻行平台创建动作
4、在动作中运行run.sh（run.sh仅适用于刻行平台运行）

在本地运行：
基本用法:
'''
python master_generate_lerobot_s.py --bag_dir <bag目录> --output_dir <输出目录> --moment_json_dir <moments.json路径> --metadata_json_dir <metadata.json路径>
'''
完整入参
```
python master_generate_lerobot_s.py \
  --bag_dir ./bags \
  --output_dir ./lerobot_output \
  --moment_json_dir ./bags/moments.json \
  --metadata_json_dir ./bags/metadata.json \
  --train_frequency 30 \
  --only_arm true \
  --which_arm both \
  --dex_dof_needed 1 # 1, 2 ,6
```
该脚本需要指定三个路径参数，与四个特征参数。
路径参数为：
需要转化的bag路径`bag_dir`：该路径下有一个bag包。
标注文件路径`moment_json_dir`：默认去`bag_dir`下找`moments.json`。
bag描述文件路径`metadata_json_dir`：默认去`bag_dir`下找`metadata.json`。

特征参数为：
转化出的lerobot数据集的帧率`train_frequency`:可选10，15，30.
当前的lerobot数据集中除了包含库帕思要求的HDF5中的所有lowdim信息，还包含乐聚用于训练act与diffusion模型的复合特征extra features:`observation.state`和`action`.
`--which_arm`:extra features考虑哪些手部数据,可选left，right，both，默认双手
`--only_arm`:extra features是否只考虑上肢和末端数据,默认是
`--dex_dof_needed`:当bag中的末端为灵巧手时，灵巧手记录的数据维度，可选1，2，6.默认1无须修改（代表只记录除大拇指外所有四指的一个共同的主动自由度的状态和动作）

## lerobot数据目录结构说明
执行脚本后，会生成标准lerobot格式数据集。目录结构如下：
```python
output_dir/                           
    ├── data/ ##Lerobotv21标准文件夹，所有的low_dim传感器状态和机器人动作数据  
    │    └──chunk-000/
    │        ├──episode_000000.parquet          
    │        ├──episode_000001.parquet   
    │        └── ...
    │
    │── meta/ ##Lerobotv21标准文件夹，lerobot生成的meta文件
    │     ├──episodes.jsonl #记录每个episode的时长            
    │     ├──info.json #记录了data/下每个parquet文件的特征信息。
    │     ├──episodes_stats.jsonl #记录了episode中特征的长度、最小值最大值、均值、方差。
    │     └──tasks.jsonl#记录当前任务名称  
    │
    ├── videos/ ##Lerobotv21标准文件夹，包含头部、双手的彩色视频，848*480的30帧mp4格式文件。
    │     └──chunk-000/ 
    │         ├──observation.images.color.head_cam_h 
    │         ├──observation.videos.color.wrist_cam_l
    │         └──observation.videos.color.wrist_cam_r
    │                  ├──episode_000000.mp4              
    │                  ├──episode_000001.mp4  
    │                  └── ...
    ├── images/ #Lerobotv21标准文件夹，其中没有数据。
    │              
    ├── depth/ #自行添加的文件夹，用于存储深度视频，包含头部、双手的16位视频，FFV1编码的848*480的30帧mkv无损视频文件
    │     └──chunk-000/
    │         ├──observation.images.color.wrist_cam_l  
    │         ├──observation.images.color.wrist_cam_r             
    │         └──observation.images.depth.head_cam_h
    │                  ├──episode_000000.mkv              
    │                  ├──episode_000001.mkv  
    │                  └── ...
    │
    │
    ├── mask/ #自行添加的文件夹，用于存储遮罩图片。遮罩图片与metadata.json中的action_config一一对应,大小与彩色图片一致。该文件仅在刻行平台运行、且记录中存在mask图片时添加。
    │     └──chunk-000/
    │         ├──episode_000000  
    │         ├──episode_000001            
    │         └──episode_000002
    │                  ├──mask_1_{action1}_{startframe1}-{endframe1}.jpg             
    │                  ├──mask_2_{action2}_{startframe2}-{endframe2}.jpg  
    │                  └── ...
    │
    ├── parameters/ #自行添加的文件夹，用于存储相机中固定不变的内参和外参数据。
    │       ├──head_cam_h_extrinsic.json  
    │       ├──head_cam_h_intrinsic.json          
    │       ├──wrist_cam_l_extrinsic.json
    │       ├──wrist_cam_l_intrinsic.json             
    │       ├──wrist_cam_r_extrinsic.json  
    │       └──wrist_cam_r_intrinsic.json
    │
    └── metadata.json #用于汇总当前lerobot数据集中各个episode的场景信息、动作信息等。


``` 
lerobot中的可用特征如下：
```python
#特征名称对应关系
dexhand = [
    "left_qiangnao_1", "left_qiangnao_2","left_qiangnao_3","left_qiangnao_4","left_qiangnao_5","left_qiangnao_6",
    "right_qiangnao_1", "right_qiangnao_2","right_qiangnao_3","right_qiangnao_4","right_qiangnao_5","right_qiangnao_6",
]
lejuclaw = [
    "left_claw", "right_claw",
]
leg=[
    "l_leg_roll", "l_leg_yaw", "l_leg_pitch", "l_knee", "l_foot_pitch", "l_foot_roll",
    "r_leg_roll", "r_leg_yaw", "r_leg_pitch", "r_knee", "r_foot_pitch", "r_foot_roll",
]
arm=[
    "zarm_l1_link", "zarm_l2_link", "zarm_l3_link", "zarm_l4_link", "zarm_l5_link", "zarm_l6_link", "zarm_l7_link",
    "zarm_r1_link", "zarm_r2_link", "zarm_r3_link", "zarm_r4_link", "zarm_r5_link", "zarm_r6_link", "zarm_r7_link",
]
head=[
    "head_yaw", "head_pitch"
]
cameras = ['head_cam_h','wrist_cam_r', 'wrist_cam_l']
imu_acc=[
        "acc_x", "acc_y", "acc_z"
    ]
imu_free_acc=[
        "free_acc_x", "ree_acc_y", "free_acc_z"
    ]
imu_gyro_acc=[
        "gyro_x", "gyro_y", "gyro_z"
    ]
imu_quat_acc=[
        "quat_x", "quat_y", "quat_z", "quat_w"
    ]
end_orientation=["left_x", "left_y", "left_z", "left_w", "right_x", "right_y", "right_z", "right_w"]
end_position=["left_x", "left_y", "left_z", "right_x", "right_y", "right_z"]
#具体可用的特征
features = {
    "action.effector.position_gripper": {"dtype": "float32", "shape": (2,),"names":lejuclaw}, #左夹爪[:,0]，右夹爪[:,1]，0 表示全开，1 表示全闭
    "action.effector.position_dexhand": {"dtype": "float32", "shape": (12,),"names":dexhand}, #左手[:, :6]，右手[:, 6:]，单位为弧度，表示目标角度
    "action.head.position": {"dtype": "float32", "shape": (2,),"names":head},#为偏航角 yaw，[:,1] 为俯仰角 pitch，单位：rad
    "action.joint.position": {"dtype": "float32", "shape": (14,), "names": arm},#关节数据 左臂[:, :7]，右臂[:, 7:]，单位为弧度
    "state.effector.position_gripper": {"dtype": "float32", "shape": (2,),"names":lejuclaw},#左[:,0]，右[:,1]，表示张开程度，单位：mm
    "state.effector.position_dexhand": {"dtype": "float32", "shape": (12,),"names":dexhand},#左[:, :6]，右[:, 6:]，关节角度，单位：rad
    "state.head.effort": {"dtype": "float32", "shape": (2,),"names":head},#电机输出扭矩
    "state.head.position": {"dtype": "float32", "shape": (2,),"names":head},#[:,0] 为偏航角 yaw，[:,1] 为俯仰角 pitch，单位：rad
    "state.head.velocity": {"dtype": "float32", "shape": (2,),"names":head},#角速度，单位：rad/s
    "state.joint.current_value": {"dtype": "float32", "shape": (14,), "names": arm},#实际关节电流
    "state.joint.effort": {"dtype": "float32", "shape": (14,), "names": arm},#实际关节扭矩
    "state.joint.position": {"dtype": "float32", "shape": (14,), "names": arm},#实际关节数据（单位：rad）
    "state.joint.velocity": {"dtype": "float32", "shape": (14,), "names": arm},#实际关节角速度（rad/s）
    "state.leg.current_value": {"dtype": "float32", "shape": (12,), "names": leg},#实际关节电流
    "state.leg.effort": {"dtype": "float32", "shape": (12,), "names": leg},#实际关节扭矩
    "state.leg.position": {"dtype": "float32", "shape": (12,), "names": leg},#实际关节数据（单位：rad）
    "state.leg.velocity": {"dtype": "float32", "shape": (12,), "names": leg},#实际关节角速度（rad/s）
    "state.end.orientation": {"dtype": "float32", "shape": (8,), "names":end_orientation},#左手与右手末端姿态四元数
    "state.end.position": {"dtype": "float32", "shape": (6,), "names":end_position},#左手与右手末端笛卡尔坐标系位置
    "imu.acc_xyz": {"dtype": "float32", "shape": (3,), "names":imu_acc},  #三轴加速度计数据
    "imu.free_acc_xyz": {"dtype": "float32", "shape": (3,), "names":imu_free_acc}, #三轴自由加速度数据
    "imu.gyro_acc_xyz": {"dtype": "float32", "shape": (3,), "names":imu_gyro_acc},  #三轴陀螺仪角速度数据
    "imu.quat_acc_xyzw": {"dtype": "float32", "shape": (4,), "names":imu_quat_acc}#本体姿态，四元数 xyzw，odom坐标系
    }
    for cam in ['head_cam_h', 'wrist_cam_r', 'wrist_cam_l']:
        features[f"observation.images.color.{cam}"] = {"dtype": "video","shape": (3, 480, 848),"names": ["channels", "height", "width"],},#三个相机彩色视频
    for cam in cameras:
	features[f"observation.camera_params.rotation_matrix_flat.{cam}"] = {#三个相机相对于基坐标系旋转矩阵
		"dtype": "float32",
		"shape": (9,),
		"names": None
	    }
	features[f"observation.camera_params.translation_vector.{cam}"] = {#三个相机相对于基坐标系平移向量
		"dtype": "float32",
		"shape": (3,),
		"names": None
	    }
```
以上为可用特征，根据参数的不同，lerobot中还可包含两个通用特征用于直接进行简单训练：
```python
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (len(DEFAULT_JOINT_NAMES_LIST_EXTRA),),
            "names": DEFAULT_JOINT_NAMES_LIST_EXTRA
        }
        features["action"] = {
            "dtype": "float32",
            "shape": (len(DEFAULT_JOINT_NAMES_LIST_EXTRA),),
            "names": DEFAULT_JOINT_NAMES_LIST_EXTRA
        }
```
`DEFAULT_JOINT_NAMES_LIST_EXTRA`可以设置不同的拼接逻辑：
1、全身（ONLY_HALF_UP_BODY=False）
如果用夹爪：
DEFAULT_ARM_JOINT_NAMES = 左臂关节 + 左夹爪 + 右臂关节 + 右夹爪
DEFAULT_JOINT_NAMES_LIST_EXTRA = 下肢关节 + DEFAULT_ARM_JOINT_NAMES + 头部关节
如果用灵巧手：
DEFAULT_ARM_JOINT_NAMES = 左臂关节 + 左 dexhand + 右臂关节 + 右 dexhand
DEFAULT_JOINT_NAMES_LIST_EXTRA = 下肢关节 + DEFAULT_ARM_JOINT_NAMES + 头部关节

2.半身（ONLY_HALF_UP_BODY=True）
如果用夹爪：
DEFAULT_ARM_JOINT_NAMES = 左臂关节 + 左夹爪 + 右臂关节 + 右夹爪
DEFAULT_JOINT_NAMES_LIST_EXTRA = 只包含DEFAULT_ARM_JOINT_NAMES拼接后的手臂关节和夹爪（不含腿和头）
如果用灵巧手：
DEFAULT_ARM_JOINT_NAMES = 左臂关节 + 左 dexhand + 右臂关节 + 右 dexhand
DEFAULT_JOINT_NAMES_LIST_EXTRA = 只包含DEFAULT_ARM_JOINT_NAMES拼接后的手臂关节和 dexhand（不含腿和头）

3.左右手选择（which_arm）
which_arm='left'：只拼接左臂关节和左夹爪/dexhand
which_arm='right'：只拼接右臂关节和右夹爪/dexhand
which_arm='both'：拼接左右臂关节和左右夹爪/dexhand

举例：
全身 + dex_hand + both :
```python
DEFAULT_JOINT_NAMES_LIST_EXTRA = [
    # 12个腿关节
    "l_leg_roll", ..., "r_foot_roll",
    # 左臂7个关节
    "zarm_l1_link", ..., "zarm_l7_link",
    # 左 dexhand 6个
    "left_qiangnao_1", ..., "left_qiangnao_6",
    # 右臂7个关节
    "zarm_r1_link", ..., "zarm_r7_link",
    # 右 dexhand 6个
    "right_qiangnao_1", ..., "right_qiangnao_6",
    # 头部2个
    "head_yaw", "head_pitch"
]
```
半身 + leju_claw + left:
```python
DEFAULT_JOINT_NAMES_LIST_EXTRA = [
    # 左臂7个关节
    "zarm_l1_link", ..., "zarm_l7_link",
    # 左夹爪1个
    "left_claw"
]
```
半身 + dex_hand + both :
```python
DEFAULT_JOINT_NAMES_LIST_EXTRA = [
    # 左臂7个关节
    "zarm_l1_link", ..., "zarm_l7_link",
    # 左 dexhand 6个
    "left_qiangnao_1", ..., "left_qiangnao_6",
    # 右臂7个关节
    "zarm_r1_link", ..., "zarm_r7_link",
    # 右 dexhand 6个
    "right_qiangnao_1", ..., "right_qiangnao_6"
]
```
## metadata.json
在lerobot数据集目录下有`metadata.json`，用于汇总当前lerobot数据集中各个episode的场景信息、动作信息等。每个episode信息如下：
```python
{
    "episode_id": "846e16d4-43e4-4c22-94f6-70ac681a5b27",
    "scene_name": "hotel services",
    "sub_scene_name": "front desk",
    "init_scene_text": "机器人站在前台后面等待客户入住",
    "english_init_scene_text": "The robots are standing behind the front desk waiting for customers to check in",
    "task_name": "客户入住",
    "english_task_name": "customer check in",
    "data_type": "常规",
    "episode_status": "approved",
    "data_gen_mode": "real_machine",
    "sn_code": "P4-202",
    "sn_name": "乐聚机器人",
    "file_duration": 14.32,#秒
    "file_size": 0.133776,#gb
    "label_info": {
        "action_config": [
            {
                "start_frame": 0,
                "end_frame": 130,
                "timestamp_utc": "2025-08-22T02:16:52.412+00:00",
                "is_mistake": false,
                "skill": "take",
                "action_text": "从人手中拿到身份证",
                "english_action_text": "take identity card from human hand"
            },
            {
                "start_frame": 130,
                "end_frame": 208,
                "timestamp_utc": "2025-08-22T02:16:57.106+00:00",
                "is_mistake": false,
                "skill": "place",
                "action_text": "将身份证放置到读卡器上",
                "english_action_text": "place identity card on card reader"
            },
            {
                "start_frame": 213,
                "end_frame": 246,
                "timestamp_utc": "2025-08-22T02:16:59.865+00:00",
                "is_mistake": false,
                "skill": "give",
                "action_text": "将身份证递给人手中",
                "english_action_text": "give identity card to human hand"
            },
            {
                "start_frame": 246,
                "end_frame": 366,
                "timestamp_utc": "2025-08-22T02:17:00.968+00:00",
                "is_mistake": false,
                "skill": "pick",
                "action_text": "从房卡盒子拿起房卡",
                "english_action_text": "pick room card from card box"
            },
            {
                "start_frame": 366,
                "end_frame": 429,
                "timestamp_utc": "2025-08-22T02:17:04.961+00:00",
                "is_mistake": false,
                "skill": "give",
                "action_text": "将房卡递给人手中",
                "english_action_text": "give room card to human hand"
            }
        ],
        "key_frame": []
    }
}

```
## 相机内外参数文件处理
对于原始数据集相机内外参数json文件，该脚本在lerobot数据集目录下新建`parameters`文件夹。内参数据包含相机的焦距x方向和y方向的缩放系数fx,fy；主点坐标ppx, ppy；畸变模型distortion_model，有 "rational_polynomial" 和 "plumb_bob"；径向畸变系数k1, k2, k3；切向畸变系数 p1, p2。
内参文件格式：
```python
{
  "intrinsic": {
    "fx": 405.2573547363281,
    "fy": 405.2271728515625,
    "ppx": 419.0584716796875,
    "ppy": 240.379638671875,
    "distortion_model": "rational_polynomial",
    "k1": -0.03300831839442253,
    "k2": 0.036516834050416946,
    "k3": -0.012877173721790314,
    "p1": -0.00010431005648570135,
    "p2": 0.0004045585519634187
  }
}
```

外参数据包含头部摄像头相对于颈关节，手部摄像头相对于腕关节的旋转矩阵rotation_matrix和平移向量translation_vector。
外参文件格式：
```python
{
    "extrinsic": {
        "rotation_matrix": [
            [
                -0.0037042639268751176,
                -0.006768463624888215,
                -0.9999702327214142
            ],
            [
                -0.0015789505765861312,
                -0.9999758072533529,
                0.006774350380800742
            ],
            [
                -0.9999918926390537,
                0.0016039975572680572,
                0.003693487241013149
            ]
        ],
        "translation_vector": [
            -0.09405207745167513,
            0.045691536635739514,
            -0.011090403668978656
        ]
    }
}
```
