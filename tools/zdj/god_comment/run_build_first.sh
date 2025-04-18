hostfile=/etc/mpi/hostfile
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=1

KWS_SERVICE_REGION=HB2
KWS_SERVICE_DC=WLF2
KWS_SERVICE_CATALOG=ai-platform.ksnserver.sparse-server
KWS_SERVICE_NAME=ai-platform-mio-kai
KWS_SERVICE_AZ=HB2AZ2
KWS_SERVICE_PAZ=HB2AZ2
KWS_SERVICE_STAGE=PROD
PYTHONPATH=.:$PYTHONPATH

# --folder viewfs://hadoop-lt-cluster/home/reco_kaiworks/dw/reco_kaiworks.db/reco_llm_quality_cmt_photo_zdj_v4_di/p_date=20250215 \
# --folder viewfs://hadoop-lt-cluster/home/reco_kaiworks/dw/reco_kaiworks.db/reco_llm_quality_cmt_photo_with_name_zdj_di/p_date=20250215 \
# --file_postfix=-c000 \
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
        python3 build_first_mpi.py \
        --folder /llm_reco_ssd/liangyiming/judge_vlm/results_two \
        --output viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zangdunju/kwai_video/Reward-Model/Val/First-Comment/Pair-Data-V2 \
        --postfix=parquet \
        --max_text_len 4096 \
        --shuffle \
        --mode="val" \
        # --debug \