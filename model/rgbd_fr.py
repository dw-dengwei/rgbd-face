from torch.nn.functional import softmax, cosine_similarity, normalize, linear
from transformers import get_polynomial_decay_schedule_with_warmup
from torchvision.models import resnet18, resnet34, resnet50
from torchmetrics.classification import Accuracy
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import CrossEntropyLoss
from model.senet import SENet
from torch.nn import Module
from torch import optim
from torch import nn

import pytorch_lightning as pl
import torchsnooper
import torch
import math


class PtlRgbdFr(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        total_steps,
        gallery,
        lr_reduce_epoch=[6, 10, 17],
        rgb_in_channels=3,
        depth_in_channels=3,
        reduction=16,
        warmup_ratio=0.05,
        backbone="resnet18",
        rgb_weight=0.5,
        arcface_margin=0.5,
        alpha=1,
        beta=2,
        gamma=3,
        lambda_1=0.5,
        lambda_2=0.05,
        momentum=0.9,
        weight_decay=0.0005,
        lr=0.1,
        out_features=1024
    ):
        super(PtlRgbdFr, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr = lr
        self.rgb_weight = rgb_weight
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.total_steps = total_steps
        self.backbone = backbone
        self.gallery = gallery
        self.lr_reduce_epoch = lr_reduce_epoch

        if "resnet" in backbone:
            self.rgbd_fr = RgbdFr(
                backbone,
                out_features=out_features
            )
        else:
            self.rgbd_fr = RgbdFr(
                backbone,
                rgb_in_channels=rgb_in_channels,
                depth_in_channels=depth_in_channels,
                reduction=reduction
            )

        self.ce_rgb = CrossEntropyLoss()
        self.sa_rgb = SemanticAlignmentLoss(beta=beta)
        self.cmfl_rgb = CrossModalFocalLoss(alpha=alpha, gamma=gamma)

        self.ce_depth = CrossEntropyLoss()
        self.sa_depth = SemanticAlignmentLoss(beta=beta)
        self.cmfl_depth = CrossModalFocalLoss(alpha=alpha, gamma=gamma)

        self.arcface = ArcFace(
            in_features=out_features,
            out_features=num_classes,
            m=arcface_margin
        )

        self.acc_rgb = Accuracy()
        self.acc_depth = Accuracy()

        self.acc_valid = Accuracy()

    def forward(self, rgb, depth):
        return self.rgbd_fr(rgb, depth)

    def training_step(self, batch, batch_idx):
        rgb, depth, label = batch
        feat_rgb, feat_depth = self(rgb, depth)
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
        self.log(
            "train/feature_sim",
            cosine_similarity(feat_rgb, feat_depth).mean(),
            **log_kwargs
        )
        self.log(
            "train/ce_rgb",
            l_cls_rgb.mean(),
            **log_kwargs
        )
        self.log(
            "train/ce_depth",
            l_cls_depth.mean(),
            **log_kwargs
        )
        self.log(
            "train/cmfl_rgb",
            l_cmfl_rgb.mean(),
            **log_kwargs
        )
        self.log(
            "train/cmfl_depth",
            l_cmfl_depth.mean(),
            **log_kwargs
        )
        self.log(
            "train/pred_true_prob_rgb",
            pred_true_prob_rgb.mean(),
            **log_kwargs
        )
        self.log(
            "train/pred_true_prob_depth",
            pred_true_prob_depth.mean(),
            **log_kwargs
        )
        self.log(
            "train/total_loss",
            l_total,
            **log_kwargs
        )
        self.log(
            "train/rgb_loss",
            l_rgb.mean(),
            **log_kwargs
        )
        self.log(
            "train/depth_loss",
            l_depth.mean(),
            **log_kwargs
        )
        self.log(
            "train/rgb_acc",
            self.acc_rgb,
            **log_kwargs
        )
        self.log(
            "train/depth_acc",
            self.acc_depth,
            **log_kwargs
        )

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
            self.gallery_feat_rgb, \
            self.gallery_feat_depth, \
            self.gallery_label = self.extract_gallery_features()

        rgb, depth, label = batch
        probe_feat_rgb, probe_feat_depth = self(rgb, depth)
        probe_feat_rgb /= probe_feat_rgb.norm(dim=1, keepdim=True)
        probe_feat_depth /= probe_feat_depth.norm(dim=1, keepdim=True)

        sim_rgb = torch.matmul(
            probe_feat_rgb,
            self.gallery_feat_rgb.T
        )
        sim_depth = torch.matmul(
            probe_feat_depth,
            self.gallery_feat_depth.T
        )
        sim_fuse = (sim_rgb + sim_depth) / 2

        pred_idx = torch.argmax(sim_fuse, dim=1)
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
        depth_list = []
        label_list = []
        for data in self.gallery:
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



class RgbdFr(Module):
    def __init__(self, backbone, *args, **kwargs):
        super(RgbdFr, self).__init__()
        if "resnet" in backbone:
            self.rgb_net = backbone_factory(
                backbone,
                out_features=kwargs["out_features"]
            )
            self.depth_net = backbone_factory(
                backbone,
                out_features=kwargs["out_features"]
            )
        else:
            self.rgb_net = backbone_factory(
                backbone,
                in_channels=kwargs["rgb_in_channels"],
                reduction=kwargs["reduction"]
            )
            self.depth_net = backbone_factory(
                backbone,
                in_channels=kwargs["depth_in_channels"],
                reduction=kwargs["reduction"]
            )

    def forward(self, rgb, depth):
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
        feat_depth = self.depth_net(depth)
        return feat_rgb, feat_depth


def backbone_factory(backbone="resnet18", *args, **kwargs):
    """
    use resnet as backbone
    Args:
        out_features: output feature size

    Returns:
        net: backbone model, the last layer is replaced
    """
    if backbone == "resnet18":
        net = resnet18()
    elif backbone == "resnet34":
        net = resnet34()
    elif backbone == "resnet50":
        net = resnet50()
    elif backbone == "senet":
        net = SENet(*args, **kwargs)
    else: # default resnet18
        net = resnet18()

    if "resnet" in backbone:
        net.fc = nn.Linear(net.fc.in_features, kwargs["out_features"])

    return net


class SemanticAlignmentLoss(Module):
    """
    semantic alignment loss
    """

    def __init__(self, beta=2):
        super(SemanticAlignmentLoss, self).__init__()
        self.beta = beta

    def forward(self, feat_m, feat_n, l_cls_m, l_cls_n):
        """
        Args:
            feat_m: feature vector m
            feat_n: feature vector n
            l_cls_m: cross entropy loss m
            l_cls_n: cross entropy loss n
        Returns:
            semantic alignment loss
        """
        rho = torch.exp(
            self.beta * (l_cls_m - l_cls_n)
            ) - 1 if l_cls_m > l_cls_n else 0
        cos = cosine_similarity(feat_m, feat_n)

        return rho * (1 - cos)


class CrossModalFocalLoss(Module):
    """
    cross modal focal loss
    """

    def __init__(self, alpha=1, gamma=3):
        super(CrossModalFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, prob_m: torch.Tensor, prob_n: torch.Tensor):
        """
        compute cross modal focal loss
        Args:
            prob_m: m-branch's probability of true class
            prob_n: n-branch's probability of true class
        Returns:
            CMFL
        """
        w = (prob_n * 2 * prob_m * prob_n) / (prob_m + prob_n)

        return -self.alpha * (1 - w) ** self.gamma * torch.log(prob_m)


class ArcFace(Module):
    def __init__(
        self, in_features, out_features, s=30.0, m=0.50, easy_margin=False
    ):
        """
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin $\cos(\theta + m)$
            easy_margin:
        """
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def update_m(self, new_m):
        self.m = new_m
        self.cos_m = math.cos(new_m)
        self.sin_m = math.sin(new_m)
        self.th = math.cos(math.pi - new_m)
        self.mm = math.sin(math.pi - new_m) * new_m

    def forward(self, feature, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = linear(normalize(feature), normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=feature.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (
                            1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output
