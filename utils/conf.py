import gc

import torch

from utils.exception import FileEmptyError


def file_empty_to_exception(f, ex_info):
    content = f.read()
    if not content:
        raise FileEmptyError(ex_info)
    return content


# 清理 设备 内存函数
def torch_gc(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc_collect()


def gc_collect():
    gc.collect()