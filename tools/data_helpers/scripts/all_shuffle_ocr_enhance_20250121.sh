hostfile=/etc/mpi/hostfile
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=4096

KWS_SERVICE_REGION=HB2
KWS_SERVICE_DC=WLF2
KWS_SERVICE_CATALOG=ai-platform.ksnserver.sparse-server
KWS_SERVICE_NAME=ai-platform-mio-kai
KWS_SERVICE_AZ=HB2AZ2
KWS_SERVICE_PAZ=HB2AZ2
KWS_SERVICE_STAGE=PROD
PYTHONPATH=.:$PYTHONPATH

mpirun --allow-run-as-root -np $np \
        -mca plm_rsh_args "-p ${Port}"  \
        -mca opal_set_max_sys_limits 1 \
        -mca plm_rsh_num_concurrent 300 \
        --oversubscribe \
        -hostfile $hostfile \
        -x PYTHONPATH=$PYTHONPATH \
        -x JAVA_HOME=$JAVA_HOME \
        -x HIVE_HOME=$HIVE_HOME \
        -x CLASSPATH=$CLASSPATH \
        -x HADOOP_USER_NAME=$HADOOP_USER_NAME \
        -x HADOOP_HOME=$HADOOP_HOME \
        -x SPARK_HOME=$SPARK_HOME \
        -x KWS_SERVICE_REGION=$KWS_SERVICE_REGION \
        -x KWS_SERVICE_DC=$KWS_SERVICE_DC \
        -x KWS_SERVICE_CATALOG=$KWS_SERVICE_CATALOG \
        -x KWS_SERVICE_NAME=$KWS_SERVICE_NAME \
        -x KWS_SERVICE_AZ=$KWS_SERVICE_AZ \
        -x KWS_SERVICE_PAZ=$KWS_SERVICE_PAZ \
        -x KWS_SERVICE_STAGE=$KWS_SERVICE_STAGE \
        python3 tools/data_helpers/all_shuffle.py \
        --buffer_mem_size 1000000000 \
        --out_partition 1024 \
        --output_dir viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2_shuffle/ocr_enhance_20250121 \
        --input_dir \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/AS10M_real \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/DenseFusion-1M \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/DenseFusion-4V-100k \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/GEOS \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Geos_QA \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/HME100K \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/IAM \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/InternVL_SAM_1B_multi_en \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/InternVL_SAM_1B_multi_zh \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/InternVL_SAM_1B_single_en \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/InternVL_SAM_1B_single_zh \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/LLaVA-CC3M-Pretrain-595K \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Mementos \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/POIE \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Recaption \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/TextCaps \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/TextCaps_All \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/VisDial_Caption \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/VisDial_Dialog \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/docvqa-spdocvqa_imdb \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/rvl-cdip \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/sharegptvideo-pretrain-video_caption_pretrain \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/synthdog \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/the_cauldron_recaption_v1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/CC12M \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/Coyo \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/ImageNet_21K \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/Laion2B_en \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/Laion2B_zh \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/Laion_Coco \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/MSCOCO \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/WebSightV1_CODE \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/WebSightV1_OCR \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/WebSightV2_CODE \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/WebSightV2_OCR \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/coco_captions \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/pdfa_ocr