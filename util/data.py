from pytorch_lightning import LightningDataModule
from config.config import BaseConfig as config
from dataset.lock3dface import Lock3DFace
from torch.utils.data import DataLoader
from dataset.texas import Texas
from dataset.vgg import Vgg
from os.path import join
from glob import glob

import pandas as pd


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
            self.dataset['valid_probe'],
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset['test_probe'],
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
    if config.using_test == 'lock3dface':
        valid_gallery, valid_probe = lock3dface_split(
            config.lock3dface_datainfo,
            config.lock3dface_data_root,
            config.lock3dface_rgb_dir,
            config.lock3dface_dzyx_dir
        )
        valid_gallery_dataset = Lock3DFace(
            valid_gallery,
            None,
            config.lock3dface_using_modal
        )
        valid_probe_dataset = Lock3DFace(
            valid_probe,
            None,
            config.lock3dface_using_modal
        )
    else:
        valid_gallery, valid_probe = texas_split(
            config.texas_data_root,
            config.texas_rgb_dir,
            config.texas_dzyx_dir,
            config.texas_ids
        )
        valid_gallery_dataset = Texas(
            *get_texas_rgb_dzyx_path(valid_gallery),
            None,
            config.texas_using_modal
        )
        valid_probe_dataset = Texas(
            *get_texas_rgb_dzyx_path(valid_probe),
            None,
            config.texas_using_modal
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
            lambda p: p.replace(dzyx_dir, rgb_dir).replace("_dzyx.png", ".jpg"),
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


def lock3dface_split(
    info_path: str,
    data_root: str,
    rgb_dir: str,
    depth_dir: str
):
    """
    :param info_path: pandas dataframe
        dir_name, ID, subset, test
        002_Kinect_NU_1,1,NU,gallery
        ...
    :param data_root:
    :param rgb_dir:
    :param depth_dir:
        data_root/
            - rgb_dir
            - depth_dir
    :return:
    """
    info = pd.read_csv(info_path, index_col="dir_name")
    rgb_fp_lst = list(
        glob(
            join(data_root, rgb_dir, "**", '*.jpg'), recursive=True
        )
    )
    gallery = []
    probe = []
    """
    [
        [rgb_fp, depth_fp, ID, subset],
        [rgb_fp, depth_fp, ID, subset]
        ...
    ]
    """

    for rgb_fp in rgb_fp_lst:
        depth_fp = rgb_fp.replace(
            rgb_dir,
            depth_dir
        ).replace(
            "RGB",
            "DEPTH"
        ).replace(
            ".jpg",
            "_depth_normal.png"
        )

        dir_name = rgb_fp.split("/")[-2].replace("RGB", "")
        sample_id = int(rgb_fp.split("/")[-1].split(".")[0])

        info_item = info.loc[dir_name]
        ID = info_item["ID"]
        subset = info_item["subset"]
        test = info_item["test"]
        if sample_id == 1 and test == "gallery":
            gallery.append([
                rgb_fp, depth_fp, ID, subset
            ])
        else:
            probe.append([
                rgb_fp, depth_fp, ID, subset
            ])

    return gallery, probe


if __name__ == "__main__":
    gallery, probe = lock3dface_split(
        config.lock3dface_datainfo,
        config.lock3dface_data_root,
        config.lock3dface_rgb_dir,
        config.lock3dface_dzyx_dir
    )
    print(probe)
