from torchvision import transforms as trans
from torch.utils.data import Dataset

import numpy as np
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
        return normal[:,:,::-1]

    @staticmethod
    def random_patch(rgb, dzyx, prob=0.2, patch_size=30):
        h, w, _ = rgb.shape
        if prob > 0 and np.random.rand() < prob:
            depth = dzyx[:, :, 0]
            normal = dzyx[:, :, 1:]
            center_r = np.random.randint(patch_size, h - patch_size)
            center_c = np.random.randint(patch_size, w - patch_size)

            left = max(0, center_c - patch_size // 2)
            right = min(w, center_c + patch_size // 2)
            bottom = max(0, center_r - patch_size // 2)
            top = min(h, center_r + patch_size // 2)

            rgb[bottom:top, left:right, :] = 0
            depth[bottom:top, left:right] = depth[bottom:top, left:right].max()
            normal = Vgg.get_normal(depth)
            dzyx = np.concatenate([depth.reshape(h, w, 1), normal], axis=2)

        return rgb, dzyx

    def __getitem__(self, index):
        label = torch.tensor(int(self.rgb_path[index].split("/")[-2]) - 1)
        out = []
        rgb = cv2.imread(self.rgb_path[index])
        dzyx = cv2.imread(self.dzyx_path[index], -1)
        rgb, dzyx = Vgg.random_patch(rgb, dzyx, prob=0, patch_size=30)
        if "rgb" in self.using_modal:
            rgb = trans.ToTensor()(rgb)
            out.append(rgb)

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
    rgb = cv2.imread('/home/dw/vgg/aligned/1/1.jpg')
    dzyx = cv2.imread('/home/dw/vgg/depth_normal/1/1_dzyx.png')
    rgb, dzyx = Vgg.random_patch(rgb, dzyx, 1.0)
    depth = dzyx[:, :, 0]
    normal = dzyx[:, :, 1:]
    cv2.imwrite('test_rgb.jpg', rgb)
    cv2.imwrite('test_dzyx.png', dzyx)
    cv2.imwrite('test_depth.jpg', depth)
    cv2.imwrite('test_normal.png', normal)
