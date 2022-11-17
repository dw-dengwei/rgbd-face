from torch.utils.data import DataLoader
from model.rgbd_fr import PtlRgbdFr
from dataset.vgg import Vgg
import pytorch_lightning as pl


dataset = Vgg('/home/dw/vgg/', 'crop', 'depth_normal', None, 468, using_modal=('rgb', 'normal'))
dataloader = DataLoader(dataset, batch_size=2)

model = PtlRgbdFr(num_classes=600)

trainer = pl.Trainer(max_epochs=10, log_every_n_steps=1)
trainer.fit(model=model, train_dataloaders=dataloader)
