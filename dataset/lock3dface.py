from torchvision import transforms as trans
from torch.utils.data import Dataset

import torch
import cv2


class Lock3DFace(Dataset):
    def __init__(self, file_lst, transform, using_modal) -> None:
        super().__init__()
        self.file_lst = file_lst
        self.using_modal = using_modal
        self.transform = transform

    def __getitem__(self, index):
        rgb_fp, dzyx_fp, ID, subset = self.file_lst[index]
        ID = torch.tensor(ID)
        out = []

        if "rgb" in self.using_modal:
            rgb = cv2.imread(rgb_fp)
            rgb = trans.ToTensor()(rgb)
            out.append(rgb)

        dzyx = cv2.imread(dzyx_fp, -1)
        if "depth" in self.using_modal:
            depth = dzyx[:, :, :1]
            depth = depth.repeat(3, axis=2)
            depth = trans.ToTensor()(depth)
            out.append(depth)
        if "normal" in self.using_modal:
            normal = dzyx[:, :, 1:]
            normal = trans.ToTensor()(normal)
            out.append(normal)

        return *out, subset, ID

    def __len__(self):
        return len(self.file_lst)
