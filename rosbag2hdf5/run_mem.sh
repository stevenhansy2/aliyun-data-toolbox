#!/bin/bash

cocli record list-moments $COS_RECORDID -o json | jq -r '{moments: .events}' >$COS_FILE_VOLUME/moments.json

# 内存测试
pip install memory_profiler matplotlib

mprof run -o run1.dat --interval 0.2 --include-children python3 cvt_rosbag2hdf5_addtopic_addframe.py \
  --bag_dir "$COS_FILE_VOLUME" \
  --moment_json_dir "$COS_FILE_VOLUME/moments.json" \
  --metadata_json_dir "$COS_FILE_VOLUME/metadata.json" \
  --output_dir "$COS_FILE_VOLUME/output" \
  --scene "test_scene" \
  --sub_scene "test_sub_scene" \
  --continuous_action "test_continuous_action" \
  --mode "simplified"

mprof peak run1.dat
mprof plot run1.dat --output run1.png
mv run1.png $COS_OUTPUT_VOLUME/
