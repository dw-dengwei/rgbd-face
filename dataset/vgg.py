from torchvision import transforms as trans
from torch.utils.data import Dataset
from config.config import BaseConfig as config

import numpy as np
import torch
import lmdb
import cv2


class Vgg(Dataset):
    def __init__(
        self,
        rgb_path,
        dzyx_path,
        seg_path,
        seg_lmdb_path,
        transform,
        using_modal=("rgb", "depth")
    ):
        super(Vgg, self).__init__()
        self.rgb_path = rgb_path
        self.dzyx_path = dzyx_path
        self.seg_path = seg_path
        self.seg_lmdb_path = seg_lmdb_path
        self.transform = transform
        self.using_modal = using_modal
        self.random_patch_prob = config.random_patch_prob
        self.random_patch_size = config.random_patch_size
        self.random_patch_method = config.random_patch_method
        self.atts = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
                     'eye_g', 'l_ear', 'r_ear',
                     'ear_r', 'nose', 'mouth', 'u_lip', 'l_lip', 'neck',
                     'neck_l',
                     'cloth', 'hair', 'hat']
        self.seg_channel = [
            0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13
        ]
        self.env = None
        self.cnt_get = 0

        assert len(self.rgb_path) == len(self.dzyx_path), (
        len(self.rgb_path), len(self.dzyx_path))
        assert len(using_modal) > 1

    def __len__(self):
        return len(self.rgb_path)

    @staticmethod
    def get_normal(depth_map):
        """calculate normal map from depth map

        Args:
            depth_map (ndarray): depth map

        Returns:
            ndarray: normal map
        """
        d_im = depth_map.astype(np.float32)
        zy, zx = np.gradient(d_im)

        normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
        n = np.linalg.norm(normal, axis=2)
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n

        # offset and rescale values to be in 0-255
        normal += 1
        normal /= 2
        # if show, comment
        normal *= 255
        # cv2.imwrite("normal.png", normal[:, :, ::-1])
        return normal[:, :, ::-1]

    def random_patch(self, rgb, dzyx):
        h, w, _ = rgb.shape
        if self.random_patch_prob > 0 and np.random.rand() < \
                self.random_patch_prob:
            depth = dzyx[:, :, 0]
            normal = dzyx[:, :, 1:]
            center_r = np.random.randint(self.random_patch_size,
                                         h - self.random_patch_size)
            center_c = np.random.randint(self.random_patch_size,
                                         w - self.random_patch_size)

            left = max(0, center_c - self.random_patch_size// 2)
            right = min(w, center_c + self.random_patch_size // 2)
            bottom = max(0, center_r - self.random_patch_size // 2)
            top = min(h, center_r + self.random_patch_size // 2)

            if self.random_patch_method == 'legacy':
                rgb[bottom:top, left:right, :] = 0
                depth[bottom:top, left:right] = 0
                normal[bottom:top, left:right] = 0
                normal = Vgg.get_normal(depth)
            else:
                rgb[bottom:top, left:right, :] = 0
                depth[bottom:top, left:right] = depth[bottom:top, left:right].max()
                normal = Vgg.get_normal(depth)
            dzyx = np.concatenate([depth.reshape(h, w, 1), normal], axis=2)

        return rgb, dzyx

    def __getitem__(self, index):
        if self.env is None:
            self.env = lmdb.open(
                self.seg_lmdb_path,
                readonly=True, lock=False, readahead=False, meminit=False
            )

        label = torch.tensor(int(self.rgb_path[index].split("/")[-2]) - 1)
        out = []
        rgb = cv2.imread(self.rgb_path[index])
        dzyx = cv2.imread(self.dzyx_path[index], -1)
        rgb, dzyx = self.random_patch(rgb, dzyx)
        if "rgb" in self.using_modal:
            # rgb = torch.from_numpy((rgb.transpose(2, 0, 1)))
            rgb = trans.ToTensor()(rgb)
            # print(rgb.shape)
            out.append(rgb)
        if "depth" in self.using_modal:
            depth = dzyx[:, :, :1]
            depth = depth.repeat(3, axis=2)
            # depth = torch.from_numpy(depth.transpose(2, 0, 1))
            depth = trans.ToTensor()(depth)
            out.append(depth)
        if "normal" in self.using_modal:
            normal = dzyx[:, :, 1:]
            # normal = torch.from_numpy(normal.transpose(2, 0, 1))
            normal = trans.ToTensor()(normal)
            out.append(normal)
        if "segment" in self.using_modal:
            id_num = self.rgb_path[index].split('/')[-2]
            file_num = self.rgb_path[index].split('/')[-1][:-4]
            with self.env.begin(write=False) as txn:
                buf = txn.get(f'{id_num}_{file_num}'.encode())
            segment = np.frombuffer(buf, dtype=np.float32).reshape(
                (11, 128, 128)
            )
            segment = torch.tensor(segment)
            out.append(segment)
            # segment = np.load(self.seg_path[index])[self.seg_channel, :, :].transpose(1,2,0)
            # segment = trans.ToTensor()(segment)
            # out.append(segment)

        return *out, label


if __name__ == '__main__':
    from util.data import get_vgg_rgb_dzyx_seg_path
    from torch.util.data import DataLoader

    train_dataset = Vgg(
        *get_vgg_rgb_dzyx_seg_path(
            config.train_data_root,
            config.train_rgb_dir,
            config.train_dzyx_dir,
            config.train_segment_dir,
            config.train_ids
        ),
        None,
        config.train_using_modal
    )
    loader = DataLoader(
        train_dataset,
        batch_size=512,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        shuffle=True
    )
