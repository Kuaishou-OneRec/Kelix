hostfile=/etc/mpi/hostfile
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=4000

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


python3 tools/data_helpers/all_shuffle_v2.py \
--world_size $np \
--make_json \
--output viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/pt/0421/stage2_ccl_v3_0425 \
--input viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/caption/Capsfusion-V3@0.5 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/opensource_data/latex-formulas@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_poie_markdown@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_poie_html@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_spdocvqa_markdown@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_sroie_markdown@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_xfund_json@1 \
        viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/HME100K@0.1 
