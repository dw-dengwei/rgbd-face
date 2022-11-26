from transformers import get_polynomial_decay_schedule_with_warmup
from model.model import ArcFace, backbone_factory
from torchmetrics.classification import Accuracy
from config.config import BaseConfig as config
from torch.nn import CrossEntropyLoss
from torch import optim

import pytorch_lightning as pl
import torch


class PtlRgbdFr(pl.LightningModule):
    def __init__(
        self,
        total_steps,
        gallery,
    ):
        super(PtlRgbdFr, self).__init__()
        """model param"""
        self.backbone = config.backbone
        self.num_classes = config.num_classes
        self.out_features = config.out_features
        self.rgb_in_channels = config.rgb_in_channels
        self.depth_in_channels = config.depth_in_channels
        self.se_reduction = config.reduction
        self.arcface_margin = config.arcface_margin
        """train param"""
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.lr = config.learning_rate
        self.warmup_steps = int(total_steps * config.warmup_ratio)
        self.total_steps = total_steps
        self.lr_reduce_epoch = config.lr_reduce_epoch
        """data param"""
        self.valid_gallery = gallery
        self.using_test = config.using_test
        self.single_modal = config.single_modal

        """initialize model"""
        self.baseline = backbone_factory(
            backbone=config.backbone,
            out_features=config.out_features
        )
        self.arcface = ArcFace(
            in_features=self.out_features,
            out_features=self.num_classes,
            m=self.arcface_margin
        )

        """initialize rgb branch loss"""
        self.ce = CrossEntropyLoss()

        """initialize training metric"""
        self.acc_train = Accuracy()
        """initialize validating(texas) metric"""
        self.acc_texas = Accuracy()
        """initialize validating(lock3dface) metric"""
        self.acc_lock3dface_avg = Accuracy()
        self.acc_lock3dface_nu = Accuracy()
        self.acc_lock3dface_fe = Accuracy()
        self.acc_lock3dface_ps = Accuracy()
        self.acc_lock3dface_oc = Accuracy()
        self.acc_lock3dface_tm = Accuracy()

    def forward(self, data):
        return self.baseline(data)

    def training_step(self, batch, batch_idx):
        rgb, depth, label = batch
        data = rgb if self.single_modal == 'rgb' else depth
        feat = self(data)
        logits = self.arcface(feat, label)

        l_cls = self.ce(logits, label)

        pred_cls = logits.argmax(dim=1)
        self.acc_train(pred_cls, label)

        log_kwargs = {
            "on_step": False,
            "on_epoch": True,
            "prog_bar": False,
            "logger": True,
            "sync_dist": True
        }
        self.log(
            "train/ce",
            l_cls.mean(),
            **log_kwargs
        )

        return l_cls

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
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval" : "step",
                "frequency": 1
            }
        }

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.valid_feat, self.valid_label = self.get_gallery_feature()

        if self.using_test == "lock3dface":
            rgb, depth, subset, label = batch
        else:
            rgb, depth, label = batch

        data = rgb if self.single_modal == 'rgb' else depth

        probe_feat = self(data)
        probe_feat /= probe_feat.norm(dim=1, keepdim=True)

        sim = torch.matmul(
            probe_feat,
            self.valid_feat.T
        )

        pred_idx = torch.argmax(sim, dim=1)
        pred_cls = self.valid_label[pred_idx]
        if self.using_test == "lock3dface":
            self.log_valid(pred_cls, label, subset)
        else:
            self.log_valid(pred_cls, label)

    def log_valid(self, pred, gt, subset=None):
        log_kwargs = {
            "on_step": False,
            "on_epoch": True,
            "prog_bar": True,
            "logger": True,
            "sync_dist": True
        }
        if self.using_test == "lock3dface":
            self.acc_lock3dface_avg(pred, gt)
            n = pred.size(0)
            for i in range(n):
                if subset[i] == 'NU':
                    self.acc_lock3dface_nu(pred[i: i + 1], gt[i: i + 1])
                elif subset[i] == 'FE':
                    self.acc_lock3dface_fe(pred[i: i + 1], gt[i: i + 1])
                elif subset[i] == 'PS':
                    self.acc_lock3dface_ps(pred[i: i + 1], gt[i: i + 1])
                elif subset[i] == 'OC':
                    self.acc_lock3dface_oc(pred[i: i + 1], gt[i: i + 1])
                elif subset[i] == 'TM':
                    self.acc_lock3dface_tm(pred[i: i + 1], gt[i: i + 1])
            self.log("valid/acc_avg", self.acc_lock3dface_avg, **log_kwargs)
            self.log("valid/acc_nu", self.acc_lock3dface_nu, **log_kwargs)
            self.log("valid/acc_fe", self.acc_lock3dface_fe, **log_kwargs)
            self.log("valid/acc_ps", self.acc_lock3dface_ps, **log_kwargs)
            self.log("valid/acc_oc", self.acc_lock3dface_oc, **log_kwargs)
            self.log("valid/acc_tm", self.acc_lock3dface_tm, **log_kwargs)
        else:
            self.acc_texas(pred, gt)
            self.log("valid/acc", self.acc_texas, **log_kwargs)

    def get_gallery_feature(self):
        data_list = []
        label_list = []
        for batch in self.valid_gallery:
            if self.using_test == "lock3dface":
                rgb, depth, subset, label = batch
            else:
                rgb, depth, label = batch
            data = rgb if self.single_modal == 'rgb' else depth

            data_list.append(data)
            label_list.append(label)

        data = torch.stack(data_list, dim=0).to(self.device)
        label = torch.stack(label_list, dim=0).to(self.device)

        gallery_feat = self(data)
        gallery_feat /= gallery_feat.norm(dim=1, keepdim=True)

        return gallery_feat, label
