import math

import torch


def get_total_train_steps(dataset, gpus, batch_size, train_epochs):
    gpus = gpus if type(gpus) == int else len(gpus)
    if gpus == -1:
        gpus = torch.cuda.device_count()

    return math.ceil(len(dataset) / gpus / batch_size) * train_epochs
