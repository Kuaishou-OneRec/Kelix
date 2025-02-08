import re

def extract_messages(data):
    """提取messages字段"""
    content = data['content']
    rst = ""
    for s in content:
        if s['type'] == 'text':
            rst += s['text']
    return rst

def extract_image(data):
    """提取image字段"""
    content = data['content']
    rst = []
    for s in content:
        if s['type'] == 'image':
            rst.append(s['image'])
    return rst

def extract_boxes_from_text(text: str) -> tuple:
    """Extract box and polygon coordinates from text"""
    boxes = []
    polygons = []
    
    box_pattern = r'<\|box_start\|>\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)<\|box_end\|>'
    box_matches = re.findall(box_pattern, text)
    
    for match in box_matches:
        x1, y1, x2, y2 = map(int, match)
        boxes.append(((x1, y1), (x2, y2)))
    
    quad_pattern = r'<\|quad_start\|>(.*?)<\|quad_end\|>'
    quad_matches = re.findall(quad_pattern, text)
    
    for match in quad_matches:
        point_pattern = r'\((\d+),\s*(\d+)\)'
        points = re.findall(point_pattern, match)
        if points:
            polygon_points = [(int(x), int(y)) for x, y in points]
            polygons.append(polygon_points)
    
    return boxes, polygons 