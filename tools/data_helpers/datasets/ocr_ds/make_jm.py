import json
import sys
import os
import base64
import datetime
import hashlib
import hmac
import requests


method = 'POST'
host = 'visual.volcengineapi.com'
region = 'cn-north-1'
endpoint = 'https://visual.volcengineapi.com'
service = 'cv'

def sign(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

def getSignatureKey(key, dateStamp, regionName, serviceName):
    kDate = sign(key.encode('utf-8'), dateStamp)
    kRegion = sign(kDate, regionName)
    kService = sign(kRegion, serviceName)
    kSigning = sign(kService, 'request')
    return kSigning

def formatQuery(parameters):
    request_parameters_init = ''
    for key in sorted(parameters):
        request_parameters_init += key + '=' + parameters[key] + '&'
    request_parameters = request_parameters_init[:-1]
    return request_parameters

def signV4Request(access_key, secret_key, service, req_query, req_body):
    if access_key is None or secret_key is None:
        print('No access key is available.')
        sys.exit()

    t = datetime.datetime.utcnow()
    current_date = t.strftime('%Y%m%dT%H%M%SZ')
    # current_date = '20210818T095729Z'
    datestamp = t.strftime('%Y%m%d')  # Date w/o time, used in credential scope
    canonical_uri = '/'
    canonical_querystring = req_query
    signed_headers = 'content-type;host;x-content-sha256;x-date'
    payload_hash = hashlib.sha256(req_body.encode('utf-8')).hexdigest()
    content_type = 'application/json'
    canonical_headers = 'content-type:' + content_type + '\n' + 'host:' + host + \
        '\n' + 'x-content-sha256:' + payload_hash + \
        '\n' + 'x-date:' + current_date + '\n'
    canonical_request = method + '\n' + canonical_uri + '\n' + canonical_querystring + \
        '\n' + canonical_headers + '\n' + signed_headers + '\n' + payload_hash
    # print(canonical_request)
    algorithm = 'HMAC-SHA256'
    credential_scope = datestamp + '/' + region + '/' + service + '/' + 'request'
    string_to_sign = algorithm + '\n' + current_date + '\n' + credential_scope + '\n' + hashlib.sha256(
        canonical_request.encode('utf-8')).hexdigest()
    # print(string_to_sign)
    signing_key = getSignatureKey(secret_key, datestamp, region, service)
    # print(signing_key)
    signature = hmac.new(signing_key, (string_to_sign).encode(
        'utf-8'), hashlib.sha256).hexdigest()
    # print(signature)

    authorization_header = algorithm + ' ' + 'Credential=' + access_key + '/' + \
        credential_scope + ', ' + 'SignedHeaders=' + \
        signed_headers + ', ' + 'Signature=' + signature
    # print(authorization_header)
    headers = {'X-Date': current_date,
               'Authorization': authorization_header,
               'X-Content-Sha256': payload_hash,
               'Content-Type': content_type
               }
    # print(headers)

    # ************* SEND THE REQUEST *************
    request_url = endpoint + '?' + canonical_querystring

    print('\nBEGIN REQUEST++++++++++++++++++++++++++++++++++++')
    print('Request URL = ' + request_url)
    try:
        r = requests.post(request_url, headers=headers, data=req_body)
    except Exception as err:
        print(f'error occurred: {err}')
        raise
    else:
        print('\nRESPONSE++++++++++++++++++++++++++++++++++++')
        print(f'Response code: {r.status_code}\n')
        # 使用 replace 方法将 \u0026 替换为 &
        resp_str = r.text.replace("\\u0026", "&")
        print(f'Response body: {resp_str}\n')
    return resp_str


import base64
import os
from PIL import Image
from io import BytesIO

def base64_to_jpg(base64_str: str, output_path: str) -> bool:
    """
    将Base64编码的字符串写入JPEG文件
    
    参数:
        base64_str: 原始Base64字符串（可能包含数据前缀）
        output_path: 输出JPEG文件的路径
    
    返回:
        成功返回True，失败返回False
    """
    try:
        # 移除可能存在的数据前缀（如"data:image/jpeg;base64,"）
        if base64_str.startswith('data:'):
            base64_str = base64_str.split(',', 1)[1]
        
        # 解码Base64字符串
        image_data = base64.b64decode(base64_str)
        
        # 验证解码后的数据是否为有效的JPEG格式
        try:
            img = Image.open(BytesIO(image_data))
            img.verify()  # 验证图像完整性
            img = Image.open(BytesIO(image_data))  # 重新打开以获取尺寸等信息
            # 确保保存为JPEG格式
            img = img.convert('RGB')
        except Exception as e:
            raise ValueError(f"无效的JPEG数据: {e}")
        
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存图像
        img.save(output_path, 'JPEG')
        
        # 验证文件是否成功写入
        if os.path.getsize(output_path) <= 0:
            raise OSError("写入的文件为空")
            
        return True
        
    except (base64.binascii.Error, ValueError, OSError) as e:
        print(f"错误: {e}")
        return False
    except Exception as e:
        print(f"未知错误: {e}")
        return False
    

'''
我需要做一个图文内容理解数据集，我需要有一个prompt调用模型绘制一个包含文字的图片。然后需要一个提供给多模态大模型的问题，以及大模型需要做出的回答。比如["制作一张vlog视频封面。马卡龙配色，美女旅游照片+色块的拼贴画风格，主文案是“威海旅游vlog”，副文案是“特种兵一日游 被低估的旅游城市”，海报主体是一个穿着短裙、梳双马尾的少女，人物白色描边","请你仔细查看给定的海报，说一下海报的内容，然后说说主文案和副文案都是什么？","海报主体是一个穿着短裙、梳双马尾的少女，主海报是'威海旅游vlog',副文案是'特种兵一日游 被低估的旅游城市'"]。一条样本就是一个三元组。请你仿照这个三元组做更多的样本，你的样本需要有多样性，然后都必须有关于文本的描述，文本可以是海报主题，也可以是书名，也可以是商铺名称等。
"活动名称是‘可持续生活节’，日期为 8 月 20 日，地点在城市公园广场。主要内容包括旧物改造工坊、植物领养、环保市集和亲子手工。"
这种不行，你还不明白吗？不可能绘制很细节的东西，P(prompt)、Q(question)和A(answer)只能有大的文字描述。不能太多文字描述，最多一句话，比如人命、书名。然后其他就是风格、或者风景、场景、人物描述（比如雪天、夏天，有一个女孩）。

你的prompt可以更丰富，比如江南撑伞女子，书名“墨雨枕书”，背景刮风下雨，池塘还有几条鱼。我说的是类似“墨雨枕书”文本相关的描述就只能一句话不能多。
'''
if __name__ == "__main__":
    import os.path as osp
    parent_dir = osp.dirname(osp.abspath(__file__))
    ds_dir = osp.join(parent_dir, 'data')
    os.makedirs(ds_dir, exist_ok=True)

    with open(f'{parent_dir}/make_jm.jsonl', 'r') as f:
        source_jsonl = f.readlines()

    source_jsonls = [json.loads(line) for line in source_jsonl]
    # https://www.volcengine.com/docs/85621/1537648
    '''
    AccessKeyId: AKLTMTc3Zjk1YWI4NDIwNGI2Y2ExYWJlMjVhYjAzMGFmOTA
    SecretAccessKey: TnpneE5ERTBPRFl6TUdZeE5EZGxObUZrT0dVek5HTTRZalF3TURWbU1UWQ==
    '''

    # 请求凭证，从访问控制申请
    access_key = 'AKLTMTc3Zjk1YWI4NDIwNGI2Y2ExYWJlMjVhYjAzMGFmOTA'
    secret_key = 'TnpneE5ERTBPRFl6TUdZeE5EZGxObUZrT0dVek5HTTRZalF3TURWbU1UWQ=='

    # 请求Query，按照接口文档中填入即可
    query_params = {
        'Action': 'CVProcess',
        'Version': '2022-08-31',
    }
    formatted_query = formatQuery(query_params)


    n_ids = len(os.listdir(ds_dir)) // 2
    for i in range(n_ids, len(source_jsonls)):

        # 请求Body，按照接口文档中填入即可
        body_params = {
            "req_key": "jimeng_high_aes_general_v21_L",
            'prompt': source_jsonls[i][0],
        }
        formatted_body = json.dumps(body_params)
        
        resp_str = signV4Request(access_key, secret_key, service,
                    formatted_query, formatted_body)
    
        with open(os.path.join(ds_dir, f'a{i}.json'), 'w') as f:
            f.write(resp_str)
        e = json.loads(resp_str)
        '''
        >>> e.keys()
        dict_keys(['code', 'data', 'message', 'request_id', 'status', 'time_elapsed'])
        >>> e['data'].keys()
        dict_keys(['algorithm_base_resp', 'binary_data_base64', 'infer_ctx', 'llm_result', 'pe_result', 'predict_tags_result', 'rephraser_result', 'request_id', 'vlm_result'])
        >>> e['data']['binary_data_base64'][0]
        '''
        base64_to_jpg(e['data']['binary_data_base64'][0], os.path.join(ds_dir, f'a{i}.jpg'))
