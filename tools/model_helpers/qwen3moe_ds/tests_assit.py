import torch


def get_assistant_mask(batch_input_ids: torch.Tensor,
                       start_pattern,
                       end_pattern):
  if not start_pattern:
    start_pattern = [151644, 77091, 198]
  if not end_pattern:
    end_pattern = [151645, 198]

  masks = []
  for input_ids in batch_input_ids:
    mask = []
    assistant_start = []
    assistant_end = []
    to_mask = False
    for _id in input_ids:
      mask.append(int(to_mask))
      if not to_mask:
        if _id in start_pattern:
          assistant_start.append(_id.item())
        else:
          assistant_start = []
        if assistant_start[-len(start_pattern):] == start_pattern:
          to_mask = True
          assistant_start = []
      else:
        print(324555, _id, end_pattern, _id in end_pattern)
        if _id in end_pattern:
          assistant_end.append(_id.item())
        else:
          assistant_end = []
        
        print(assistant_end[-len(end_pattern):] == end_pattern, 23454)
        if assistant_end[-len(end_pattern):] == end_pattern:
          to_mask = False
          print(assistant_end, 4343)
          assistant_end = []
          
    masks.append(mask)
  return torch.tensor(masks) 


def test_get_assistant_mask():
    # 测试输入
    input_ids = torch.tensor([151643,   2610,    525,    264,  10950,  17847,     13, 151669,   4340,
            525,    498,     30, 151670,  63716,     11,   9702,    498,     13,
         151645, 151669,   4340,    525,    498,     30, 151670,  63716,     11,
           9702,    498,     13, 151645]).unsqueeze(0)  # 添加batch维度
    # input_ids = torch.tensor([])
    # 自定义开始和结束模式
    start_pattern = [151670]
    end_pattern = [151645]
    
    # 调用函数
    masks = get_assistant_mask(batch_input_ids=input_ids, 
                              start_pattern=start_pattern, 
                              end_pattern=end_pattern)
    
    # 打印输入和输出
    print("输入token ID序列:")
    print(input_ids.squeeze().tolist())
    print("\n生成的掩码:")
    print(masks.squeeze().tolist())
    

# 执行测试
test_get_assistant_mask()