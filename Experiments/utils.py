import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_args(args):
    """
    打印参数对象的所有参数和对应的值
    """
    # 将args转换为字典
    args_dict = vars(args)

    # 遍历字典并打印键值对
    for key, value in args_dict.items():
        print(key, "=", value)