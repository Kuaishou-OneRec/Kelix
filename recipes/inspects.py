import torch 
import torch.nn as nn


def num_params(model):
    return sum([x.numel() for x in model.parameters()])


def info_params_recursive(model, name="", max_depth=5, curr_depth=0):
    """
    from torchvision import models
    print(info_params_recursive(models.resnet18()))
    """
    res = ""
    if curr_depth == 0:
        res += "下面每行的格式为:\n当前深度-<模型类型>(模型名称): 参数数量\t\tp0:第一个参数名:第一个参数均值\n"
    if curr_depth == max_depth: return ""
    #
    indent = '--' * (curr_depth + 1)
    named_params = list(model.named_parameters())
    if len(named_params):
        pname, pparam = sorted(named_params)[0]
        pparam = pparam.detach().mean().item()
    else:
        pname, pparam = None, None
    res += "{} {}-{}({}): {}\t\tp0:{}:{}\n".format(indent, curr_depth, type(model), name, num_params(model), pname, pparam)
    for name, model in model.named_children():
        if isinstance(model, nn.Module):
            res += info_params_recursive(model, name, max_depth, curr_depth + 1)
    return res





def format_dict_or_list(obj, indent_level=0, indent_size=2):
    """
    格式化打印dict/list，用来替代json.dumps
    """
    def format_value(value, indent_level=0, indent_size=2):
        if isinstance(value, (dict, list)):
            return format_dict_or_list(value, indent_level, indent_size)
        elif isinstance(value, str):
            return f'"{value}"'
        else:
            return str(value)

    if isinstance(obj, dict):
        items = [f": {format_value(v, indent_level + 1)}" for k, v in obj.items()]
        keys = [f'"{k}"' for k in obj.keys()]
        formatted_items = ',\n'.join(f'{(" " * indent_size * (indent_level + 1))}{k}{v}' for k, v in zip(keys, items))
        return '{\n' + formatted_items + '\n' + (' ' * indent_size * indent_level) + '}'
    elif isinstance(obj, list):
        items = [format_value(item, indent_level + 1) for item in obj]
        formatted_items = ',\n'.join(' ' * indent_size * (indent_level + 1) + item for item in items)
        return '[\n' + formatted_items + '\n' + (' ' * indent_size * indent_level) + ']'
    else:
        return obj

        
import torchvision.models as models
def _test_info_params_recursive():
    net = models.resnet18(False)
    print(
        info_params_recursive(net, max_depth=5)
    )


if __name__=='__main__':
    _test_info_params_recursive()
