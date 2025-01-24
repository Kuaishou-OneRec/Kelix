#!/bin/bash

function list_hdfs_tree {
  local path=$1
  local prefix=$2
  local depth=$3

  # List the contents of the current directory
  hadoop fs -ls "$path" | tail -n +2 | while read -r line; do
    # Extract the file or directory name
    name=$(echo "$line" | awk '{print $NF}')
    # Remove the path prefix to get the relative name
    relative_name=${name#$path/}

    # Print the current file or directory
    echo "${prefix}├── $relative_name"

    # If it's a directory and depth is less than 2, recursively list its contents
    if [[ $line == d* ]] && [[ $depth -lt 2 ]]; then
      list_hdfs_tree "$name" "$prefix│   " $((depth + 1))
    fi
  done
}

# Start listing from the specified directory with initial depth 0
start_path="viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/"
list_hdfs_tree "$start_path" "" 0