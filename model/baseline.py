from transformers import get_polynomial_decay_schedule_with_warmup
from model.rgbd_fr import backbone_factory, ArcFace
from torchmetrics.classification import Accuracy
from torch.nn import CrossEntropyLoss
from torch.nn import Module
from torch import optim

import pytorch_lightning as pl
import torch


class PtlBaseline(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        total_steps,
        gallery,
        lr_reduce_epoch=[6, 10, 17],
        warmup_ratio=0.05,
        backbone="resnet18",
        arcface_margin=0.5,
        momentum=0.9,
        weight_decay=0.0005,
        lr=0.1,
        out_features=1024
    ):
        super(PtlBaseline, self).__init__()

        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr = lr
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.total_steps = total_steps
        self.backbone = backbone
        self.gallery = gallery
        self.lr_reduce_epoch = lr_reduce_epoch

        self.rgbd_fr = RgbdFr(
            backbone,
            out_features=out_features
        )

        self.ce_rgb = CrossEntropyLoss()
        self.arcface = ArcFace(
            in_features=out_features,
            out_features=num_classes,
            m=arcface_margin
        )

        self.acc_rgb = Accuracy()

        self.acc_valid = Accuracy()

    def forward(self, rgb):
        return self.rgbd_fr(rgb)

    def training_step(self, batch, batch_idx):
        rgb, depth, label = batch
        feat_rgb = self(rgb)
        logits_rgb = self.arcface(feat_rgb, label)

        l_cls_rgb = self.ce_rgb(logits_rgb, label)

        pred_cls_rgb = logits_rgb.argmax(dim=1)
        self.acc_rgb(pred_cls_rgb, label)

        log_kwargs = {
            "on_step"  : False,
            "on_epoch" : True,
            "prog_bar" : False,
            "logger"   : True,
            "sync_dist": True
        }
        self.log(
            "train/ce_rgb",
            l_cls_rgb.mean(),
            **log_kwargs
        )
        self.log(
            "train/rgb_acc",
            self.acc_rgb,
            **log_kwargs
        )

        return l_cls_rgb

    def configure_optimizers(self):
        opt = optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # opt = optim.SGD(
        #     params=self.parameters(),
        #     lr=self.lr,
        #     weight_decay=self.weight_decay,
        #     momentum=0.9
        # )

        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            opt,
            self.warmup_steps,
            self.total_steps,
            lr_end=1e-7,
            power=4
        )
        # lr_scheduler = MultiStepLR(
        #     opt,
        #     self.lr_reduce_epoch,
        #     0.1
        # )

        return {
            "optimizer"   : opt,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval" : "step",
                "frequency": 1
            }
        }

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.gallery_feat_rgb, \
            self.gallery_label = self.extract_gallery_features()

        rgb, depth, label = batch
        probe_feat_rgb = self(rgb)
        probe_feat_rgb /= probe_feat_rgb.norm(dim=1, keepdim=True)

        sim_rgb = torch.matmul(
            probe_feat_rgb,
            self.gallery_feat_rgb.T
        )

        pred_idx = torch.argmax(sim_rgb, dim=1)
        pred_cls = self.gallery_label[pred_idx]
        self.acc_valid(pred_cls, label)
        self.log(
            "valid/acc",
            self.acc_valid,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

    def extract_gallery_features(self):
        rgb_list = []
        label_list = []
        for data in self.gallery:
            rgb, depth, label = data

            rgb_list.append(rgb)
            label_list.append(label)

        rgb = torch.stack(rgb_list, dim=0).to(self.device)
        label = torch.stack(label_list, dim=0).to(self.device)

        gallery_feat_rgb = self(rgb)
        gallery_feat_rgb /= gallery_feat_rgb.norm(dim=1, keepdim=True)

        return gallery_feat_rgb, label


class RgbdFr(Module):
    def __init__(self, backbone, *args, **kwargs):
        super(RgbdFr, self).__init__()
        self.rgb_net = backbone_factory(
            backbone,
            out_features=kwargs["out_features"]
        )

    def forward(self, rgb):
        """
        get rgb and depth branches' feature vectors
        Args:
            rgb: rgb image
            depth: depth image

        Returns:
            feat_rgb: rgb feature vector
            feat_depth: depth feature vector
        """
        feat_rgb = self.rgb_net(rgb)
        return feat_rgb
