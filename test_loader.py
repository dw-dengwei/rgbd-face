from util.data import get_vgg_rgb_dzyx_seg_path
from dataset.vgg import Vgg
from config.config import BaseConfig as config
from tqdm import tqdm
from torch.utils.data import DataLoader


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
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
    shuffle=True
)

# assert len(train_dataset) == len(set(train_dataset))
print(len(train_dataset))

for batch in tqdm(loader):
    ...

# for data in tqdm(train_dataset):
#     ...
