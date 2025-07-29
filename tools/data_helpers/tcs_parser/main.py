import struct
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

def parse_tcs_file(file_path: str) -> Dict[str, Any]:
    """
    解析 CASIA-HWDB-T 数据集的 .tcs 文件格式
    
    Args:
        file_path: .tcs 文件的路径
        
    Returns:
        包含文件所有解析信息的字典，包括文件头信息和字符图像记录
    """
    result = {
        "header": {},
        "records": []
    }
    
    try:
        with open(file_path, 'rb') as f:
            # 解析文件头
            header_size = struct.unpack('l', f.read(4))[0]
            result["header"]["size"] = header_size
            
            format_code = f.read(8).decode('ascii').strip('\x00')
            if format_code != "tcs":
                raise ValueError(f"文件格式错误，不是有效的 .tcs 文件 (找到: {format_code})")
            result["header"]["format_code"] = format_code
            
            # 计算说明文字的长度（总头大小 - 固定头大小）
            fixed_header_size = 4 + 8 + 20 + 2 + 2  # Size of Header, Format code, Code type, Code Length, Bits per pixel
            illustr_length = header_size - fixed_header_size
            illustration = f.read(illustr_length).decode('ascii').strip('\x00')
            result["header"]["illustration"] = illustration
            
            code_type = f.read(20).decode('ascii').strip('\x00')
            result["header"]["code_type"] = code_type
            
            code_length = struct.unpack('h', f.read(2))[0]
            result["header"]["code_length"] = code_length
            
            bits_per_pixel = struct.unpack('h', f.read(2))[0]
            result["header"]["bits_per_pixel"] = bits_per_pixel
            
            # 解析字符串图像记录
            while True:
                # 检查是否到达文件末尾
                if f.tell() >= header_size + struct.calcsize('h') * 5:  # 简单检查，实际应更严谨
                    break
                
                record = {}
                
                # 读取笔画宽度和行高
                sw = struct.unpack('h', f.read(2))[0]
                record["stroke_width"] = sw
                
                lh = struct.unpack('h', f.read(2))[0]
                record["line_height"] = lh
                
                ntp = struct.unpack('h', f.read(2))[0]
                record["touching_points_count"] = ntp
                
                # 读取接触点位置
                touching_points = []
                for _ in range(ntp):
                    # 每个接触点有顶部和底部端点
                    top_row = struct.unpack('h', f.read(2))[0]
                    top_col = struct.unpack('h', f.read(2))[0]
                    bottom_row = struct.unpack('h', f.read(2))[0]
                    bottom_col = struct.unpack('h', f.read(2))[0]
                    
                    touching_points.append({
                        "top": (top_row, top_col),
                        "bottom": (bottom_row, bottom_col)
                    })
                record["touching_points"] = touching_points
                
                # 读取字符数量
                nc = struct.unpack('h', f.read(2))[0]
                record["characters_count"] = nc
                
                # 读取字符标签
                label_bytes = f.read(nc * code_length)
                if code_type == "ASCII":
                    labels = [chr(byte) for byte in label_bytes]
                else:  # GB 编码
                    # 注意：实际处理 GB 编码可能需要更复杂的方式
                    labels = [label_bytes[i:i+2].decode('gb2312', errors='ignore') for i in range(0, len(label_bytes), code_length)]
                record["character_labels"] = labels
                
                # 读取图像高度和宽度
                h = struct.unpack('h', f.read(2))[0]
                w = struct.unpack('h', f.read(2))[0]
                record["image_height"] = h
                record["image_width"] = w
                
                # 读取位图数据
                bitmap_size = h * w
                bitmap_data = f.read(bitmap_size)
                # 将位图数据转换为 numpy 数组（灰度图像）
                record["bitmap"] = np.frombuffer(bitmap_data, dtype=np.uint8).reshape(h, w)
                
                result["records"].append(record)
                
        return result
    
    except Exception as e:
        print(f"解析 .tcs 文件时出错: {str(e)}")
        return result

def display_tcs_info(tcs_data: Dict[str, Any]) -> None:
    """显示解析后的 .tcs 文件信息"""
    print("=== .tcs 文件解析信息 ===")
    print(f"文件头大小: {tcs_data['header']['size']} 字节")
    print(f"格式代码: {tcs_data['header']['format_code']}")
    print(f"说明: {tcs_data['header']['illustration']}")
    print(f"编码类型: {tcs_data['header']['code_type']}")
    print(f"编码长度: {tcs_data['header']['code_length']} 字节")
    print(f"每像素位数: {tcs_data['header']['bits_per_pixel']}")
    print(f"记录数量: {len(tcs_data['records'])}")
    
    if tcs_data['records']:
        first_record = tcs_data['records'][0]
        print("\n=== 第一条记录信息 ===")
        print(f"笔画宽度: {first_record['stroke_width']}")
        print(f"行高: {first_record['line_height']}")
        print(f"接触点数量: {first_record['touching_points_count']}")
        print(f"字符数量: {first_record['characters_count']}")
        print(f"字符标签: {first_record['character_labels']}")
        print(f"图像尺寸: {first_record['image_height']}x{first_record['image_width']}")
        print(f"位图形状: {first_record['bitmap'].shape}")

# 示例用法
if __name__ == "__main__":
    # 替换为实际的 .tcs 文件路径
    file_path = "/llm_reco/lingzhixin/recovlm_qw0510/recovlm/tools/data_helpers/tcs_parser/all_Tch_ct-allGB7356-allLetter.tcs"
    tcs_data = parse_tcs_file(file_path)
    display_tcs_info(tcs_data)