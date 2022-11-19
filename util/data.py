from pytorch_lightning import LightningDataModule
from config.config import BaseConfig as config
from torch.utils.data import DataLoader
from dataset.texas import Texas
from dataset.vgg import Vgg
from os.path import join
from glob import glob


class Loader(LightningDataModule):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'],
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset['probe'],
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            shuffle=False
        )


def get_dataset():
    train_dataset = Vgg(
        *get_vgg_rgb_dzyx_path(
            config.train_data_root,
            config.train_rgb_dir,
            config.train_dzyx_dir,
            config.train_ids
        ),
        None,
        config.train_using_modal
    )

    gallery, probe = texas_split(
        config.valid_data_root,
        config.valid_rgb_dir,
        config.valid_dzyx_dir,
        config.valid_ids
    )
    valid_gallery_dataset = Texas(
        *get_texas_rgb_dzyx_path(gallery),
        None,
        config.valid_using_modal
    )
    valid_probe_dataset = Texas(
        *get_texas_rgb_dzyx_path(probe),
        None,
        config.valid_using_modal
    )

    return train_dataset, \
           valid_gallery_dataset, \
           valid_probe_dataset


def get_vgg_rgb_dzyx_path(data_root, rgb_dir, dzyx_dir, num_id):
    ids = [i for i in range(1, 1 + num_id)]
    dzyx_pattern = "*_dzyx.png"
    dzyx_list = []
    for idx in ids:
        dzyx_list.extend(
            glob(join(data_root, dzyx_dir, str(idx), dzyx_pattern))
        )

    rgb_list = list(
        map(
            lambda p: p.replace(dzyx_dir, rgb_dir).replace("_dzyx.png", ".png"),
            dzyx_list
        )
    )

    return rgb_list, dzyx_list


def get_texas_rgb_dzyx_path(test_set):
    rgb_list = []
    dzyx_list = []
    for fp in test_set:
        if fp is None:
            continue
        rgb_fp, dzyx_fp = fp
        rgb_list.append(rgb_fp)
        dzyx_list.append(dzyx_fp)

    return rgb_list, dzyx_list


def texas_split(data_root, rgb_dir, dzyx_dir, num_id):
    """
    :param data_root: Texas root
    :param rgb_dir: rgb image dir name
    :param dzyx_dir: dzyx image dir name
    :param num_id: number of ids
    :return: gallery [[rgb_fp, dzyx_fp], ..., None, ... ](len = number of ids),
             probe [[rgb_fp, dzyx_fp], ... ](len = remains)
    Note that some subjects x, y, ... has no data. Thus, gallery[x] is None
    """
    ids = [[] for i in range(num_id)]
    rgb_list = glob(join(data_root, rgb_dir, "*.png"))

    for rgb_fp in rgb_list:
        id_idx = int(rgb_fp.split("/")[-1].split("_")[-1].split(".")[0]) - 1
        dzyx_fp = rgb_fp.replace(rgb_dir, dzyx_dir).replace(".png", "_dzyx.png")
        ids[id_idx].append(
            [rgb_fp, dzyx_fp]
        )

    gallery = [None] * num_id
    probe = []
    for i in range(num_id):
        if len(ids[i]) == 0:
            continue
        gallery[i] = ids[i][0]
        test = ids[i][1:]
        if len(test) != 0:
            probe.extend(test)

    return gallery, probe


if __name__ == "__main__":
    gallery, probe = texas_split("/home/dw/rgbd/Texas", "aligned",
                                 "depth_normal", 118)
    g_rgb, g_dzyx = get_texas_rgb_dzyx_path(gallery)
    p_rgb, p_dzyx = get_texas_rgb_dzyx_path(probe)
