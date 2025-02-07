#!/bin/bash

# base_dir="viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/single_image"
# dirs=("Captioning" "General_QA" "Mathematics" "Chart" "OCR" "Knowledge" "Document" "Grounding" "Science" "Conversation" "Medical" "GUI")

# for dir in "${dirs[@]}"; do
#   hadoop fs -mkdir "${base_dir}/${dir}"
# done

# base_dir="viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/multi_images"
# dirs=("Document" "General_QA")

# for dir in "${dirs[@]}"; do
#   hadoop fs -mkdir "${base_dir}/${dir}"
# done

# base_dir="viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/multi_images"
# dirs=("Document" "General_QA")

# for dir in "${dirs[@]}"; do
#   hadoop fs -mkdir "${base_dir}/${dir}"
# done

# base_dir="viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/video"
# dirs=("Captioning" "General_QA" "GUI")

# for dir in "${dirs[@]}"; do
#   hadoop fs -mkdir "${base_dir}/${dir}"
# done

base_dir="viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/text"
dirs=("General_QA" "Code" "Long_Context" "Mathematics" "Knowledge")

for dir in "${dirs[@]}"; do
  hadoop fs -mkdir "${base_dir}/${dir}"
done