hostfile=/etc/mpi/hostfile
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=100

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
        --output_dir viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/shuffle/20250326_sft \
        --input_dir \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/sft_caption@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/sft_summary@1  \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/sft_summary_by_section@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/sft_msy_caption@0.1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/search_after_watch@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zangdunju/kwai_video/Continue-SFT/v0.2/Format@0.02 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zangdunju/kwai_video/Continue-SFT/v0.2/Quality@0.02	\
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zangdunju/kwai_video/Continue-SFT/v0.2/Reason@0.02 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zangdunju/kwai_video/Continue-SFT/v0.2/Session@0.02 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zangdunju/kwai_video/Continue-SFT/v0.2/Type@0.02 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/sharegptvideo-sft-video_240k_caption_15k@0.03

