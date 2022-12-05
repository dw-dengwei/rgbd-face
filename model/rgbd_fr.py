from transformers import get_polynomial_decay_schedule_with_warmup
from torch.nn.functional import softmax, cosine_similarity
from torchmetrics.classification import Accuracy
from torch.nn import CrossEntropyLoss
from model.model import (
    ArcFace, CrossModalFocalLoss, SemanticAlignmentLoss,
    RgbdFr
)
from torch import optim
from config.config import BaseConfig as config

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
        """loss param"""
        self.lambda_1 = config.lambda_1
        self.lambda_2 = config.lambda_2
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.rgb_weight = config.rgb_weight
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

        """initialize model"""
        self.rgbd_fr = RgbdFr(self.out_features)
        self.arcface = ArcFace(
            in_features=self.out_features,
            out_features=self.num_classes,
            m=self.arcface_margin
        )

        """initialize rgb branch loss"""
        self.ce_rgb = CrossEntropyLoss()
        self.sa_rgb = SemanticAlignmentLoss(beta=self.beta)
        self.cmfl_rgb = CrossModalFocalLoss(alpha=self.alpha, gamma=self.gamma)
        """initialize depth branch loss"""
        self.ce_depth = CrossEntropyLoss()
        self.sa_depth = SemanticAlignmentLoss(beta=self.beta)
        self.cmfl_depth = CrossModalFocalLoss(alpha=self.alpha, gamma=self.gamma)

        """initialize training metric"""
        self.acc_rgb = Accuracy()
        self.acc_depth = Accuracy()
        """initialize validating(texas) metric"""
        self.acc_texas = Accuracy()
        """initialize validating(lock3dface) metric"""
        self.acc_lock3dface_avg = Accuracy()
        self.acc_lock3dface_nu = Accuracy()
        self.acc_lock3dface_fe = Accuracy()
        self.acc_lock3dface_ps = Accuracy()
        self.acc_lock3dface_oc = Accuracy()
        self.acc_lock3dface_tm = Accuracy()

    def forward(self, rgb, depth, segment=None):
        return self.rgbd_fr(rgb, depth, segment)

    def training_step(self, batch, batch_idx):
        rgb, depth, segment, label = batch
        feat_rgb, feat_depth = self(rgb, depth, segment)
        logits_rgb = self.arcface(feat_rgb, label)
        logits_depth = self.arcface(feat_depth, label)

        l_cls_rgb = self.ce_rgb(logits_rgb, label)
        l_cls_depth = self.ce_depth(logits_depth, label)

        pred_true_prob_rgb = torch.take_along_dim(
            softmax(logits_rgb, dim=1),
            label.unsqueeze(1),
            dim=1
        )
        pred_true_prob_depth = torch.take_along_dim(
            softmax(logits_depth, dim=1),
            label.unsqueeze(1),
            dim=1
        )

        pred_cls_rgb = logits_rgb.argmax(dim=1)
        pred_cls_depth = logits_depth.argmax(dim=1)
        self.acc_rgb(pred_cls_rgb, label)
        self.acc_depth(pred_cls_depth, label)

        l_sa_rgb = self.sa_rgb(feat_rgb, feat_depth, l_cls_rgb, l_cls_depth)
        l_sa_depth = self.sa_depth(feat_depth, feat_rgb, l_cls_depth, l_cls_rgb)

        l_cmfl_rgb = self.cmfl_rgb(pred_true_prob_rgb, pred_true_prob_depth)
        l_cmfl_depth = self.cmfl_depth(pred_true_prob_depth, pred_true_prob_rgb)

        l_rgb = (1 - self.lambda_1) * l_cls_rgb + \
                self.lambda_1 * l_cmfl_rgb + \
                self.lambda_2 * l_sa_rgb

        l_depth = (1 - self.lambda_1) * l_cls_depth + \
                  self.lambda_1 * l_cmfl_depth + \
                  self.lambda_2 * l_sa_depth

        l_total = (self.rgb_weight * l_rgb + (1 - self.rgb_weight) * l_depth).mean()

        log_kwargs = {
            "on_step": False,
            "on_epoch": True,
            "prog_bar": False,
            "logger": True,
            "sync_dist": True
        }
        log_content = {
            "train/feature_sim": cosine_similarity(feat_rgb, feat_depth).mean(),
            "train/ce_rgb": l_cls_rgb.mean(),
            "train/ce_depth": l_cls_depth.mean(),
            "train/cmfl_rgb": l_cmfl_rgb.mean(),
            "train/cmfl_depth": l_cmfl_depth.mean(),
            "train/pred_true_prob_rgb": pred_true_prob_rgb.mean(),
            "train/pred_true_prob_depth": pred_true_prob_depth.mean(),
            "train/total_loss": l_total,
            "train/rgb_loss": l_rgb.mean(),
            "train/depth_loss": l_depth.mean(),
            "train/rgb_acc": self.acc_rgb,
            "train/depth_acc": self.acc_depth,
            "train/l_sa_rgb": l_sa_rgb.mean(),
            "train/l_sa_depth": l_sa_depth.mean(),
        }
        self.log_dict(log_content, **log_kwargs)

        return l_total

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
            self.valid_feat_rgb, \
            self.valid_feat_depth, \
            self.valid_label = self.get_gallery_feature()

        if self.using_test == "lock3dface":
            rgb, depth, subset, label = batch
        else:
            rgb, depth, label = batch

        probe_feat_rgb, probe_feat_depth = self(rgb, depth)
        probe_feat_rgb /= probe_feat_rgb.norm(dim=1, keepdim=True)
        probe_feat_depth /= probe_feat_depth.norm(dim=1, keepdim=True)

        sim_rgb = torch.matmul(
            probe_feat_rgb,
            self.valid_feat_rgb.T
        )
        sim_depth = torch.matmul(
            probe_feat_depth,
            self.valid_feat_depth.T
        )
        sim_fuse = (sim_rgb + sim_depth) / 2

        pred_idx = torch.argmax(sim_fuse, dim=1)
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

            log_content = {
              "valid/acc_avg": self.acc_lock3dface_avg,
              "valid/acc_nu": self.acc_lock3dface_nu,
              "valid/acc_fe": self.acc_lock3dface_fe,
              "valid/acc_ps": self.acc_lock3dface_ps,
              "valid/acc_oc": self.acc_lock3dface_oc,
              "valid/acc_tm": self.acc_lock3dface_tm,
            }
            self.log_dict(log_content, **log_kwargs)
        else:
            self.acc_texas(pred, gt)
            self.log("valid/acc", self.acc_texas, **log_kwargs)

    def get_gallery_feature(self):
        rgb_list = []
        depth_list = []
        label_list = []
        for data in self.valid_gallery:
            if self.using_test == "lock3dface":
                rgb, depth, subset, label = data
            else:
                rgb, depth, label = data

            rgb_list.append(rgb)
            depth_list.append(depth)
            label_list.append(label)

        rgb = torch.stack(rgb_list, dim=0).to(self.device)
        depth = torch.stack(depth_list, dim=0).to(self.device)
        label = torch.stack(label_list, dim=0).to(self.device)

        gallery_feat_rgb, gallery_feat_depth = self(rgb, depth)
        gallery_feat_rgb /= gallery_feat_rgb.norm(dim=1, keepdim=True)
        gallery_feat_depth /= gallery_feat_depth.norm(dim=1, keepdim=True)

        return gallery_feat_rgb, gallery_feat_depth, label
