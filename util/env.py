import numpy as np
import random
import torch
import os


def seed_all(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)  # 为了禁止hash随机化，使得实验可复现。

    torch.manual_seed(random_seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed_all(random_seed)  # 为所有GPU设置随机种子（多块GPU）

    torch.backends.cudnn.deterministic = True
