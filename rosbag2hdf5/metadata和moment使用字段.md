moment中使用的字段
```python
{
  "moments": [
    {
        "mark_start":"2025-09-06 17:47:41.526", #现在标注平台中具有的字段，开始时间
        "mark_end":"2025-09-06 17:47:57.644", #现在标注平台中具有的字段，结束时间

        "triggerTime": "2025-09-19T05:47:04.424Z",#现在使用的刻行平台的一刻字段，动作开始时间，不需要
        "duration": "15.763",#现在使用的刻行平台的一刻字段，动作持续时间，不需要

        "en_desc": "Initial position",
        "skill_detail": "移动到货架初始位置",
        "start_position": "0.6315789473624994",
        "mark_type": "step",
        "end_position": "0.9862938931162807",
        "skill_atomic_en": "move",
        "en_skill_detail": "move to storage rack",
        "ch_desc": "初始位置",
    },
  ]
}
```
```python
{
  "device_sn": "P4-210",
  "eef_type": "leju_claw",
  "scene_zh_name": "制造工厂",
  "scene_zh_dec": "",
  "scene_code": "manufacturing plant",
  "sub_scene_zh_dec": "大件或小件工装放置货架上，机器人位于货架前",
  "sub_scene_code": "loading of tooling",
  "task_group_code": "loading_of_small_tooling",
  "task_code": "XJRW_jz",
  "sub_scene_zh_name": "工装上料",
  "scene_en_dec": "manufacturing plant",
  "task_group_name": "小件工装上料",
  "sub_scene_en_dec": "Large or small loads placed on the shelf and robots located in front of the shelf",
  "task_name": "小件工装上料-夹爪"
}
```
