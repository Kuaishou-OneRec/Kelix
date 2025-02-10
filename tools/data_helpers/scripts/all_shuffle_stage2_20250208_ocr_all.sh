hostfile=/etc/mpi/hostfile
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=600

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
        --output_dir viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2_shuffle/stage2_20250208_ocr_all \
        --input_dir \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/RenderedText_chat@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/LatexOCR_orig@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/LatexOCR_Rendered@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/LatexFormul@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/kwai_cover@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/DenseFusion@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/SyntheticOCR_EN_HW@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/SyntheticOCR_CN_HW@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_infovqa_json@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_infovqa_ocr@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_poie_json@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_poie_markdown@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_spdocvqa_json@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/SpDocVQA_IMDB@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_spdocvqa_markdown@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_sroie_json@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_sroie_markdown@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_xfund_html@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_xfund_json@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_xfund_markdown@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_table_qa_merge@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_poie_html@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_sroie_html@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Docmatrix@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/WebSightV2_CHAT@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/WebSightV2_CODE@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/WebSightV2_OCR@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/PlotQA_Multi_Turn@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/ChartQA@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/the_cauldron@0.1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/the_cauldron_recaption_v1@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/HME100K@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_grounding/TextOCR_grounding@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_grounding/LSVT_grounding@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_grounding/ArT_grounding@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_grounding/COCOText_grounding@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_grounding/MLT_grounding@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_grounding/MTWI_grounding@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_grounding/ReCTs_grounding@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_grounding/SROIE_grounding@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_grounding/ArT_grounding@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/pubtabnet@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/fintabnet@1 \
viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/chart_recaptions