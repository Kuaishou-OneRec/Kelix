#!/bin/bash

# 检查是否提供了目录参数
if [ -z "$1" ]; then
  echo "Usage: $0 <HDFS_DIRECTORY>"
  exit 1
fi

# HDFS 目录
HDFS_DIR="$1"

# 临时文件存储需要删除的文件路径
temp_file=$(mktemp)

# 获取目录下所有文件及其大小
hadoop fs -ls -R $HDFS_DIR | awk '{if ($5 < 1024) print $5, $8}' | while read -r size file; do
  # 输出当前文件和大小
  echo "Marking for deletion: $file, Size: $size bytes"
  # 将文件路径添加到临时文件中
  echo "$file" >> "$temp_file"
done

# 读取临时文件中的文件路径
if [ -s "$temp_file" ]; then
  files_to_delete=$(cat "$temp_file")
  echo "The following files will be deleted:"
  echo "$files_to_delete"
  
  # 用户确认
  read -p "Are you sure you want to delete these files? (y/n): " confirm
  if [ "$confirm" = "y" ]; then
    echo "Deleting files..."
    hadoop fs -rm $files_to_delete
  else
    echo "Deletion cancelled."
  fi
else
  echo "No files to delete."
fi

# 删除临时文件
rm "$temp_file"