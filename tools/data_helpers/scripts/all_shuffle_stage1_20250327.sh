hostfile=/etc/mpi/hostfile
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=500

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
        --buffer_mem_size 21474836480 \
        --output_dir viewfs://hadoop-lt-cluster/home/reco_wl/mpi/maosiyang/shuffle/20250327_mix_v1 \
        --input_dir \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/asr_text@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/asr_text_v1@1\
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/asr_interleave@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/multiocr@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/video_ocr_Qwen@1\
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/kwai_detail_caption_mp4@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/detail_caption_2_5@1\
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/maosiyang/kwai_video/click_after_show_top10/train/v1@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/maosiyang/kwai_video/title_caption_category/train/v1@0.10 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zangdunju/kwai_video/Continue-Pretrain/First-Comment/Only-Quality-All@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zangdunju/kwai_video/Continue-Pretrain/Session-Comment-Reply-Weight-Multi@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/maosiyang/kwai_video/query_title_caption/train/v1@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/maosiyang/kwai_video/shuffle/train/v3@1\
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/sharegptvideo-pretrain-video_caption_pretrain@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zangdunju/kwai_video/Continue-Pretrain/Outer/Wanzhi@1\
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/sharegptvideo-sft-video_240k_caption_15k@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/text/General_QA/infinity_instruct_7m@0.5 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhouyang12/web_comment/l1@1\
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2_shuffle/stage2_20250208_ocr_all@0.20
