from torchvision import transforms as trans
from torch.utils.data import Dataset
from os.path import join
from glob import glob
import torch
import cv2


class Vgg(Dataset):
    def __init__(
        self,
        data_root,
        rgb_dir,
        dzyx_dir,
        transform,
        num_id,
        using_modal=("rgb", "depth")
    ):
        super(Vgg, self).__init__()
        self.rgb_path, self.dzyx_path = get_rgb_dzyx_path(data_root, rgb_dir, dzyx_dir, num_id)

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


def get_rgb_dzyx_path(data_root, rgb_dir, dzyx_dir, num_id):
    ids = [i for i in range(1, 1 + num_id)]
    dzyx_pattern = '*_dzyx.png'
    dzyx_list = []
    for idx in ids:
        dzyx_list.extend(
            glob(join(data_root, dzyx_dir, str(idx), dzyx_pattern))
        )

    rgb_list = list(
        map(lambda p: p.replace(dzyx_dir, rgb_dir).replace('_dzyx.png', '.png'), dzyx_list)
    )

    return rgb_list, dzyx_list


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    vgg = Vgg('/home/wei/remote/vgg/', 'crop', 'depth_normal', None, 468)
    loader = DataLoader(vgg)
    for idx, (rgb, depth, label) in enumerate(loader):
        print(rgb, depth, label)
        break
