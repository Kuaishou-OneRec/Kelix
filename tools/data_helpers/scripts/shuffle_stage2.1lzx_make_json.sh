

hostfile=/etc/mpi/hostfile
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=6500

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
        --output viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm_0427compile/recovlm3/recovlm/tools/data_helpers/scripts/shuffle_stage2.1lzx \
		--input viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/caption/Capsfusion-V3@0.05 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/opensource_data/latex-formulas@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_poie_markdown@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_poie_html@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_spdocvqa_markdown@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_sroie_markdown@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_xfund_json@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/HME100K@0.1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/IAM@0.1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/EST_VQA@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/opensource_data/st-vqa@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/InfoVQA_OCR@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/SROIE@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/POIE@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_grounding/ArT_grounding@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/LSVT_grounding_OCR@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/RCTW@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_grounding/ReCTs_grounding@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/MTWI@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/TextVQA_OCR@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/CASIA@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/coco_captions@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/TextCaps@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/MMInstruct-Caption-EN@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/MMInstruct-Caption-ZH@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/InternVL_SAM_1B_multi_en@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/InternVL_SAM_1B_multi_zh@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/caption/ShareGPT4V@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/llavar@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/OCR_VQA@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/TextOCR_fix@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/DenseFusion@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/the_cauldron_recaption_v1@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/Refcoco@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/V3Det@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/RenderedText_chat@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/SyntheticOCR_CN_HW@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/SyntheticOCR_EN_HW@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/WebSightV2_OCR@0.3 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/InternVL_SAM_1B_single_en@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/InternVL_SAM_1B_single_zh@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/ASV2@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/GRIT@0.3 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/DataComp@0.03 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/Coyo@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/Laion2B_en@0.005 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/Laion_Coco@0.005 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/COCOText@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/fintabnet@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/opensource_data/parsynth-ocr-200k@1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/synthdog_fix@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/caption/GRIT@0.15 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/caption/Detailed_Caption_densecap_new@1 \
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
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/MMC4_ff_reconstruct@0.02	 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Obelisc_reconstruct@0.02 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_wanjuan3lang/wanjuan3lang/train_v1@0.01 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_wanjuan_mm_image/wanjuan_mm_image/train_v1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Wanjuan_reconstruct@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_wikihow_vgsi/wikihow_vgsi/train_v1 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/OpenR1Verified8k \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Mementos \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-web/train_v1@0.5 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-qa/train_v1@0.4 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-text-code-block/train_v1@0.005 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-translated-code/train_v1@0.3 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-web-pro/train_v1@0.35 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_original/text/General_QA/infinity_instruct_7m@0.533 \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/liangyiming/instruct_data \
                viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/HME100K




