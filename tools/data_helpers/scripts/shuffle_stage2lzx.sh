

hostfile=/etc/mpi/hostfile
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=5000

mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pip3 install fastparquet==2024.2.0"
mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pip3 install humanize"


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
        python3 tools/data_helpers/all_shuffle_v2.py \
        --output viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm_0427compile/recovlm3/recovlm/tools/data_helpers/scripts/shuffle_stage2lzx \
        --input viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/caption/Detailed_Caption_densecap_new@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/GQA \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/opensource_data/OK-VQA	\
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/opensource_data/A-OKVQA \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/visual7w \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/vsr \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/opensource_data/mm_tallyqa	\
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Objects365 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/opensource_data/ICON-QA \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/VisDial_Dialog \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/opensource_data/VQAv2 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/hateful_memes	\
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_infovqa_json \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_infovqa_ocr \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/SpDocVQA_IMDB	\
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_table_qa_merge \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/LLaVA-OneVision \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/ChartQA \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/caption/PlotQA \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/caption/ArxivQA	\
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/caption/TabMwp \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/caption/DVQA \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/uni-chart \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/caption/Chart2Text \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/fintabnet \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/DocReason25K	\
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/lmms_lab_DocVQA \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/SujetFinanceVisionQA \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/SujetFinanceVisionOCR \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Mavis_Function \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Mavis_Geometry_EN	\
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Mavis_Geometry_EN_QA_fix \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Mavis_Geometry_QA_EN \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Mavis_Geometry_QA_ZH \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Mavis_Geometry_ZH \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Mavis_Geometry_ZH_QA_fix \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/GeomVerse \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/MapQA \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Geos_QA \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Geometry3K \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/GEOS \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Clevr-Math@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/mathematics/MathV360K \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/mathematics/cmm-math \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/mathematics/DeepMath-103K \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_xfund_html \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_xfund_markdown \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_poie_html \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_sroie_html \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/WebSightV2_CODE \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/PMC_VQA \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/VQA_RAD \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/SLAKE	\
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/MMC4_ff_reconstruct@0.35	 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Obelisc_reconstruct@0.36 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_wanjuan3lang/wanjuan3lang/train_v1@0.25 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_wanjuan_mm_image/wanjuan_mm_image/train_v1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Wanjuan_reconstruct@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_wikihow_vgsi/wikihow_vgsi/train_v1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/OpenR1Verified8k \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Mementos \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-web/train_v1@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-qa/train_v1@0.4 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-text-code-block/train_v1@0.3 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-translated-code/train_v1@0.3 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-web-pro/train_v1@0.35 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/text/General_QA/infinity_instruct_7m@0.533 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/liangyiming/instruct_data 



