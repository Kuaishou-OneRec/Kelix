# Step1: download using recovlm.services.clients.PidInfoClient

# Input: a file of pid list, seperate by \n

# for each pid, download via:
# PidInfoClient().get_pid_info(1468086843497)
# return:
# {
#   "error": None,
#   "media_path": /path/,
#   "pid": 1468086843497,
#   "success": True,
#   "text_fields": {
#     "asr": "",
#     "caption": "",
#     "ocr": "",
#     "text": "",
#     "title": ""
#   },
#   "timing": {
#     "media_retrieval": 0.3490316867828369,
#     "text_retrieval": 3.919825315475464,
#     "total": 4.268903017044067
#   }
# }

# Step2: prepare dataset
# prepare dataset as a parquet, 
# format;

# 000000000.json
# {
#     "__key__": 000000000, 
#     "messages": [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "video",
#                     "video": "file:///path/to/video1.mp4",
#                     "max_pixels": 360 * 420,
#                     "fps": 1.0,
#                     "video_start": 0,
#                     "video_end":
#                 },
#                 {"type": "text", "text": "Describe this video."},
#             ]
#         },
#         {"role": "assistant", "content": "The video describe ..."},
#     ],
#     "source": "kwai_video"
# }

# Step3: run batch infer
# call recovlm/recipes/offline_batch_inference.py