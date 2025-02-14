nohup python3 chatgpt_client.py gpt_config_with_cmt_cot.yaml > output1.log 2>&1 &
nohup python3 chatgpt_client.py gpt_config_with_cmt_no_cot.yaml > output2.log 2>&1 &
nohup python3 chatgpt_client.py gpt_config_with_cot_no_cmt.yaml > output3.log 2>&1 &
nohup python3 chatgpt_client.py gpt_config_without_cmt_cot.yaml > output4.log 2>&1 &
