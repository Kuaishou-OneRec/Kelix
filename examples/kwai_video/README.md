# 站内视频有关

这个目录下提供一些准备站内视频数据的脚本。

### 功能1: 根据photo_id列表下载视频信息

适合小批量数据的下载（<10w）。首先准备pid list，写到.txt文件中，用`\n`分割，比如
```
1527823123
1512312313
```
切换到项目根目录，然后运行
`bash examples/kwai_video/run_download.sh /path/of/pids /path/of/cache`

视频信息会被下载到指定的cache，注意**非RecoVLM项目组**同学记得切换路径，否则下载内容可能会被删除。

### 功能2: 将下载的视频整理成数据集，用来做推理。

也适合小批量数据的处理（<10w）。同样准备pid list，写到.txt文件中，用`\n`分割。

然后运行`bash examples/kwai_video/run_dataset.sh /path/of/pids output_dir /path/of/photo`

其中`output_dir`是你的输出路径，`/path/of/photo`是存放pid info的路径，如果使用了`run_download.sh`脚本准备数据，默认情况下是`/path/of/cache/Photo`