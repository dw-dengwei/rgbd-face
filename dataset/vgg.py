from torchvision import transforms as trans
from torch.utils.data import Dataset

import torch
import cv2


class Vgg(Dataset):
    def __init__(
        self,
        rgb_path,
        dzyx_path,
        transform,
        using_modal=("rgb", "depth")
    ):
        super(Vgg, self).__init__()
        self.rgb_path = rgb_path
        self.dzyx_path = dzyx_path
        self.transform = transform
        self.using_modal = using_modal

        assert len(self.rgb_path) == len(self.dzyx_path), (len(self.rgb_path), len(self.dzyx_path))
        assert len(using_modal) > 1

    def __len__(self):
        return len(self.rgb_path)

    def __getitem__(self, index):
        label = torch.tensor(int(self.rgb_path[index].split("/")[-2]) - 1)
        out = []
        if "rgb" in self.using_modal:
            rgb = cv2.imread(self.rgb_path[index])
            rgb = trans.ToTensor()(rgb)
            out.append(rgb)

        dzyx = cv2.imread(self.dzyx_path[index], -1)
        if "depth" in self.using_modal:
            depth = dzyx[:, :, :1]
            depth = trans.ToTensor()(depth)
            out.append(depth)
        if "normal" in self.using_modal:
            normal = dzyx[:, :, 1:]
            normal = trans.ToTensor()(normal)
            out.append(normal)

        return *out, label


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    vgg = Vgg('/home/wei/remote/vgg/', 'crop', 'depth_normal', None, 468)
    loader = DataLoader(vgg)
    for idx, (rgb, depth, label) in enumerate(loader):
        print(rgb, depth, label)
        break
