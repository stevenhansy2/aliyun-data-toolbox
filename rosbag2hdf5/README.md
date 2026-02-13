# rosbag2HDF5

## 准备工作

### 镜像准备

刻行中的自动化可以使用公开镜像或者上传到刻行镜像仓库中的镜像，在本地或者 CI 构建完镜像后，可以上传到刻行的镜像仓库，文档可以详见
[准备镜像](https://docs.coscene.cn/docs/image/build-image)

如在本地测试可以用

```
docker build -t leju:latest . # 本地 Build
docker run -it --rm -v ./:/workspace/leju leju:latest bash  # 本地执行，并 Mount 本目录到指定目录，方便测试

# 在容器中
# export 相关变量

./run.sh
```

### 数据准备

确保记录（Record）中，已有如下文件

- Bag 文件
- 对应的 metadata.json 文件

### 刻行 Action 运行环境相关 - coScene Action Runtime

#### Action 中注入的环境变量

刻行会在 Action 中注入如下环境变量，您可以在 Action 中使用这些环境变量，方便编程，API，CLI 等工具使用

- `$COS_FILE_VOLUME`: 刻行当前装载的 Records 文件在 $COS_FILE_VOLUME 下，注意在脚本中使用环境变量替代绝对路径
- `$COS_RECORDID`: 当前挂载的记录的 ID
- `$COS_PROJECTID`：当前所在的项目的 ID

#### coCLI

为了在镜像中获取一些额外的信息，或者上传下载文件到新记录，或者其他项目，您可以使用 coCLI。`Dockerfile` 中已经加入了 coCLI 的安装步骤

在刻行的 Action 中，所有运行 coCLI 的凭证信息都已经配置完成，可以直接使用

## 脚本变量总览

- `TARGET_PROJECT_SLUG`：是否将转换完的数据，上传到项目中创建记录
- `CREATE_NEW_RECORD`：如果不是新的项目，是否在同项目中创建记录
- `SUCCESS_ADDITIONAL_LABELS`：是否要添加额外的成功标签

## Action 执行流程

详细的流程控制见 `run.sh` 脚本，以下是流程描述

1. 从所在的 Record 中，使用刻行 CLI 工具获取相关 Moments，存入临时文件夹
2. 通过脚本，将 Bag + metadata + moments 数据，转换为最终的 HDF5 文件形式
3. 根据脚本的运行环境变量配置，执行如下可能的上传动作

   - 上传到【新项目】的【新记录】
   - 上传到【当前项目】的【新记录】
   - 上传到【当前记录】的指定文件目录下

4. 根据 Action 的运行环境变量配置，给当前记录和新记录打上标签

### 1. 获取 Moments

rosbag2HDF5 需要 moments 的信息输入，需要从刻行中获取 Moments 文件，可以使用 coCLI 配合环境变量获取当前 Records 的
moments.json 文件

我们将 moments 文件存入到 COS_FILE_VOLUME 中，也就是 Records 数据所在目录，方便后续使用

```bash
cocli record list-moments $COS_RECORD_ID -o json | jq -r '{moments: .events}' >$COS_FILE_VOLUME/moments.json
```

### 2. 进行转换

准备好我们的 3 样数据，bag + metadata + moments 之后，就可以开始实际的转换。

注意到我们这里对输入输出的目录加入了一些变量，来控制行为，可以根据实际需要灵活修改。

这里我们从 Records 文件夹中读入所有数据，并将结果写回 Records 文件夹中的 Output 路径。

```bash
python3 cvt_rosbag2hdf5.py \
 --bag_dir "$COS_FILE_VOLUME" \
 --moment_json_dir "$COS_FILE_VOLUME/moments.json" \
 --metadata_json_dir "$COS_FILE_VOLUME/metadata.json" \
  --output_dir "$COS_FILE_VOLUME/output" \
 --scene "test_scene" \
 --sub_scene "test_sub_scene" \
 --continuous_action "test_continuous_action" \
 --mode "simplified"
```

### 3. 判断失败还是成功

#### 3.1 失败

如果失败，终止程序，会给原记录打上标签，默认失败标签为

```
r25:failed
```

#### 3.2 成功

如果转换成功，根据配置信息，决定上传行为，并打上对应的标签。关键环境变量如下，优先级从下而上

- `TARGET_PROJECT_SLUG`：目标项目的 SLUG
- `CREATE_NEW_RECORD`：是否创建新记录

另外如果操作成功时，默认会打入如下成功标签。

```
r25:success
```

如果在 Action 配置中检测到

```
SUCCESS_ADDITIONAL_LABELS # 标签 String 数组，多个的话用逗号分开，如 123,456
```

则会在成功的记录上，打入额外的成功标签。

##### 如果配置了 `TARGET_PROJECT_SLUG`

则程序会在【目标项目】中创建对应记录，并将转换结果上传到该记录中，打上对应标签后，退出程序。

新记录的模型默认为 【原记录名字 + 转换结果】

##### 如果没配置 `TARGET_PROJECT_SLUG`，且配置了 `CREATE_NEW_RECORD`

则会在【当前项目】中创建新的记录，并将转换结果上传到该记录，打上对应标签后，退出程序

新记录的模型默认为 【原记录名字 + 转换结果】

##### 如果前俩变量都没配置，则会将文件上传到本记录中，默认放置在 `output` 文件夹下

## Benchmark

使用了常见的 Bag 数据进行测试，结果如下

1. Bag 包大小，163.96 MB，内存占用 242.508 MiB

![bag-mem-usage](./image/bag-mem-usage.png)
