import pandas as pd

df1 = pd.read_parquet('/home/leju_kuavo/kuavo_cb/temp/Kuavo-Data-Tool-Box-test-memory/Kuavo-Data-Tool-Box-test-memory/rosbag2lerobot/testoutput/aa4bff56-26e5-41ad-b25a-29a33230a7e1/aa4bff56-26e5-41ad-b25a-29a33230a7e1/aa4bff56-26e5-41ad-b25a-29a33230a7e1/data/chunk-000/episode_000000.parquet')          # 默认 engine='pyarrow'
print(df1.keys())

df2 = pd.read_parquet('/home/leju_kuavo/kuavo_cb/temp/Kuavo-Data-Tool-Box-test-memory/Kuavo-Data-Tool-Box-test-memory/rosbag2lerobot/testoutput/aa4bff56-26e5-41ad-b25a-29a33230a7e1/aa4bff56-26e5-41ad-b25a-29a33230a7e1/aa4bff56-26e5-41ad-b25a-29a33230a7e1/data/chunk-000/episode_000000.parquet')
# print(df2[["timestamp","frame_index","episode_index","index","task_index"]].head())
print(df2.columns)