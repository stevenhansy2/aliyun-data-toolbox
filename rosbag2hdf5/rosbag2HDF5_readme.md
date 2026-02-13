# rosbag2HDF5 脚本使用说明
## 1.运行环境
目前使用标准lerobot的python环境运行。后续可以精简转换环境所需的包。
运行程序为cvt_rosbag2hdf5.py，不要改动其余的py文件和request.json（该json制定了py中处理rosbag的部分逻辑。）
## 2.脚本功能
### 该脚本用于生成标准库帕斯HDF5数据集目录。
```
<target_dir>/
├── task_info/    //用于存放每个场景+子场景的 JSON 描述文件，命名格式为：场景名称-子场景名称-连续动作名称.json
│   ├── Ward-Measuring_Temperature-check_thermometer.json
│   └── ...
└── Ward-53p21GB_2000counts_85p30h/    //以场景命名，格式为: 场景-大小(GB)_条数_时长，例如文件夹名表示场景为 Ward，数据体积为 53.21 GB，共 2000 条记录，持续时间为 85.30 小时（"p" 表示小数点）
    └── Measuring_Temperature-53p21GB_2000counts_85p30h/   //子场景目录，格式同上
        └── check_thermometer-53p21GB_2000counts_85p30h/   //连续动作目录，格式同上
            ├── UUID1/    //以 UUID 命名，对应 `task_info` 中的 `episode_id` 字段。同时，每个 UUID 目录下包括 `camera/`（图像）、`parameters/`（相机参数）、 `proprio_stats/`（动作数据）和 `audio/`（音频数据）四部分
            │   ├── camera/
            │   │   ├── depth/
            │   │   │   └── depth.mkv  //建议使用 ffv1 无损压缩
            │   │   └── video/
            │   │       ├── hand_right_color.mp4    //mp4必须为h264格式
            │   │       └── ...
            │   ├── parameters/            //对应视角的相机内外参(名字前缀与video中相机名保持一致， 且以下仅为展示，不是包含以下3个即可)
            │   │   ├── hand_right_extrinsic_params.json
            │   │   ├── hand_right_extrinsic_params_aligned.json
            │   │   ├── hand_right_intrinsic_params.json
            │   │   └──...
            │   ├── proprio_stats/       
            │   │   └── proprio_stats.hdf5    //整个连续动作的HDF5 文件（不需要拆分为原子技能hdf5）
            │   └──audio
            │       └──microphone.wav         //wav格式
            └── UUID2/    //以 UUID 命名，格式同上
```
### 也可用于生成单个rosbag包的简化目录：
```
            ├── UUID1/    //以 UUID 命名，对应 `task_info` 中的 `episode_id` 字段。同时，每个 UUID 目录下包括 `camera/`（图像）、`parameters/`（相机参数）、 `proprio_stats/`（动作数据）和 `audio/`（音频数据）四部分
            │   ├── camera/
            │   │   ├── depth/
            │   │   │   └── depth.mkv  //建议使用 ffv1 无损压缩
            │   │   └── video/
            │   │       ├── hand_right_color.mp4    //mp4必须为h264格式
            │   │       └── ...
            │   ├── parameters/            //对应视角的相机内外参(名字前缀与video中相机名保持一致， 且以下仅为展示，不是包含以下3个即可)
            │   │   ├── hand_right_extrinsic_params.json
            │   │   ├── hand_right_extrinsic_params_aligned.json
            │   │   ├── hand_right_intrinsic_params.json
            │   │   └──...
            │   ├── proprio_stats/       
            │   │   └── proprio_stats.hdf5    //整个连续动作的HDF5 文件（不需要拆分为原子技能hdf5）
            │   └──audio
            │       └──microphone.wav         //wav格式
            
```
## 3.运行方式
### 读取数据要求：
该脚本通过具有单个rosbag、单个metadata.json与单个moment.json的特定文件夹。在指定路径输出转化后的数据。
### 运行示例：
```python
python cvt_rosbag2hdf5.py --bag_dir "./testbag/" --moment_json_dir "./testbag/moment.json" --metadata_json_dir "./testbag/metadata.json" --output_dir "./testbag/" --scene "test_scene" "--sub_scene" "test_sub_scene" --continuous_action "test_continuous_action" --mode "simplified"
```
### 参数说明：
`--bag_dir`: 必须传入。用于指定读取的文件夹。该文件夹下应该有用于转化的唯一rosbag。例如 `"./testbag"`

`--moment_json_dir`:可选参数。用于指定当前任务的时刻说明文件。例如`"./testbag/moment.json"`如不指定，默认从`--bag_dir`下读取`moment.json"`。

`--metadata_json_dir`:可选参数。用于指定当前任务的原始信息。例如`"./testbag/metadata.json"`如不指定，默认从`--bag_dir`下读取`metadata.json"`。

`--output_dir`:必须传入。用于指定输出目录所在的文件夹。

`--scene`:可选参数，用于指定当前任务场景。默认`"test_scene" `.

`--sub_scene`:可选参数，用于指定当前子任务场景。默认`"test_sub_scene" `.

`--continuous_action`:可选参数，用于指定当前任务连续动作名。默认`test_continuous_action`.

`--mode`:可选参数，可选`complete`或`simplified`。用于指定生成完整目录或简化目录。默认`simplified`。

## 4.运行结果
运行结果后，根据`--mode`参数设置的不同，生成指定目录与内容。
主要生成文件内容如下：
### (1) 说明文件
`metadata_merge.json`:对单个rosbag数据的说明。例如：
```json
{
    "episode_id": "93330343-ee08-48b8-ae60-e6add74cc5b7",
    "scene_name": "Ward",
    "sub_scene_name": "Measuring_Temperature",
    "init_scene_text": "一个机器人站在黑体前",
    "english_init_scene_text": "A robot stands in fornt of a balck body.",
    "task_name": "检查校准体温计",
    "english_task_name": "check_thermometer",
    "data_type": "常规",
    "episode_status": "approved",
    "data_gen_mode": "real_machine",
    "sn_code": "A2D0001AB00029",
    "sn_name": "宇树",
    "label_info": {
        "action_config": [
            {
                "start_frame": 90,
                "end_frame": 200,
                "timestamp_utc": "2025-06-27T17:43:06.388+08:00",
                "skill": "",
                "action_text": "结束抓取物件",
                "english_action_text": ""
            },
            {
                "start_frame": 30,
                "end_frame": 80,
                "timestamp_utc": "2025-06-27T17:42:58.533+08:00",
                "skill": "",
                "action_text": "开始抓取物件",
                "english_action_text": ""
            }
        ],
        "key_frame": []
    }
}
```
### (2) 深度视频
以mkv格式，FFV1编码保存的深度视频。默认包含左、右手、头三个摄像头的视频。

### (3) 视频
以mp4格式保存的图像视频。默认包含左、右手、头三个摄像头的视频。

### （4）摄像头参数信息
以json格式保存的三个摄像头内参信息。目前的rosbag摄像头参数话题中只包含该内参。内部有D，K，R，P四个参数。
```json
{
  "D": [
    -0.03169431909918785,
    0.03369959816336632,
    0.00017958383250515908,
    -0.00011798791820183396,
    -0.011600286699831486
  ],
  "K": [
    0.0,
    0.0,
    0.0,
    366.3092346191406,
    0.0,
    320.86151123046875,
    0.0,
    366.36468505859375,
    243.1219024658203
  ],
  "R": [
    0.0,
    0.0,
    1.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0
  ],
  "P": [
    0.0,
    0.0,
    1.0,
    366.3092346191406,
    0.0,
    320.86151123046875,
    0.0,
    0.0,
    366.36468505859375,
    243.1219024658203,
    0.0,
    0.0
  ]
}
```

### （5）HDF5文件
该文件记录了机器人各部分的动作和传感器参数。对rosbag中没有的数据进行了标注，已有数据格式按照库帕斯要求。
```python
/ (根目录)
├── timestamps                      
├── action/                          
│   ├── effector/ 
│   │   ├── index                  
│   │   ├── position (gripper)     
│   │   └── position (dexhand)      
│   ├── end/ #无末端数据 标记为NaN
│   │   ├── orientation             //(N,2,4) float32 // 左末端[:,0,:]，右末端[:,1,:]，flange四元数 [x,y,z,w]
│   │   ├── position                //(N,2,3) float32 // 左末端[:,0,:]，右末端[:,1,:]，flange xyz, 单位：米
│   │   └── index                   //(M,) int64 // 与 control source 控制信号时间对齐的索引
│   ├── head/
│   │   ├── position                //(N,2) float32 // [:,0] 为偏航角 yaw，[:,1] 为俯仰角 pitch，单位：rad
│   │   └── index                   //(M,) int64 // 控制信号时间索引
│   ├── joint/
│   │   ├── position                //(N,14) float32 //关节数据 左臂[:, :7]，右臂[:, 7:]，单位为弧度
│   │   └── index                   //(M,) int64 // 控制信号时间索引
│   ├── robot/ #无robot的action 标记为NaN
│   │   ├── velocity                //(N,2) float32 // [:,0] 表示 x 方向速度，[:,1] 表示航向角速度（yaw rate）
│   │   └── index                   //(M,) int64 // 控制信号时间索引
│   └── waist/ #无腰部数据，标记为NaN
│       ├── position                //(N,2) float32 // [:,0] 为俯仰 pitch，[:,1] 为升降 lift，单位：rad 和 m
│       └── index                   //(M,) int64 // 控制信号时间索引
├── state/                            // 机器人传感器观测到的状态
│   ├── effector/ 
│   │   ├── force                   //(N,2) float32 // 左右夹爪/灵巧手的受力 #无受力 标记为NaN
│   │   └── position (gripper)     //(N,2) float32 // 左[:,0]，右[:,1]，表示张开程度，单位：mm
│   │   └── position (dexhand)     //(N,12) float32 // 左[:, :6]，右[:, 6:]，关节角度，单位：rad
│   ├── end/ #无末端数据 标记为NaN
│   │   ├── angular                //(N,2,3) float32 // 角速度 [wx, wy, wz]，单位：rad/s
│   │   ├── orientation            //(N,2,4) float32 // 左[:,0,:]，右[:,1,:]，flange四元数 xyzw
│   │   ├── position               //(N,2,3) float32 // 左[:,0,:]，右[:,1,:]，flange xyz, 单位：米
│   │   ├── velocity               //(N,2,3) float32 // 空间速度 [vx, vy, vz]，单位：m/s
│   │   └── wrench                 //(N,2,6) float32 // [{fx,fy,fz,mx,my,mz}，{}] 左右末端的六维力，无则为空
│   ├── head/
│   │   ├── effort                 //(N,2) float32 // 电机输出扭矩
│   │   ├── position               //(N,2) float32 // [:,0] 为偏航角 yaw，[:,1] 为俯仰角 pitch，单位：rad
│   │   └── velocity               //(N,2) float32 // 角速度，单位：rad/s
│   ├── joint/
│   │   ├── current_value          //(N,14) float32 // 左臂[:, :7]，右臂[:, 7:]，如为电流等内部数据 #无电流
│   │   ├── effort                 //(N,14) float32 // 实际关节扭矩
│   │   ├── position               //(N,14) float32 // 实际关节数据（单位：rad）
│   │   └── velocity               //(N,14) float32 // 实际关节角速度（rad/s）
│   ├── robot/		
│   │   ├── orientation            //(N,4) float32 // 本体姿态，四元数 xyzw，odom坐标系
│   │   ├── orientation_drift     //(N,4) float32 // odom->map dirft 四元数 世界坐标系 #无orientation_drift 标记为NaN
│   │   ├── position               //(N,3) float32 // 机体位置 {odom}系 #无position 标记为NaN
│   │   └── position_drift        //(N,3) float32 // odom->map dirft 世界坐标系 #无position_drift 标记为NaN
│   └── waist/ #无腰部数据，标记为NaN
│       ├── effort                //(N,2) float32 // 实际腰部⼒矩
│       ├── position              //(N,2) float32 // pitch [:,0] 为角度（rad），lift [:,1] 为升降高度（m）
│       └── velocity              //(N,2) float32 // 腰部速度 
└── imu/
    ├── acc_xyz //(N,3) float32 
    ├── free_acc_xyz //(N,3) float32 
    ├── gyro_xyz //(N,3) float32 
    └── quat_xyzw //(N,4) float32 
    
            

```
