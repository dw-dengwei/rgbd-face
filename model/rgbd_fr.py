from torch.nn.functional import softmax, cosine_similarity, normalize, linear
from torchvision.models import resnet18
from torch.nn import CrossEntropyLoss
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
        alpha=1,
        beta=2,
        gamma=3,
        lambda_1=0.5,
        lambda_2=0.05,
        momentum=0.9,
        weight_decay=0.0005,
        lr=0.1,
        out_features=1024
    ) -> None:
        super(PtlRgbdFr, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr = lr

        self.rgbd_fr = RgbdFr(out_features=out_features)

        self.ce_rgb = CrossEntropyLoss()
        self.sa_rgb = SemanticAlignmentLoss(beta=beta)
        self.cmfl_rgb = CrossModalFocalLoss(alpha=alpha, gamma=gamma)

        self.ce_depth = CrossEntropyLoss()
        self.sa_depth = SemanticAlignmentLoss(beta=beta)
        self.cmfl_depth = CrossModalFocalLoss(alpha=alpha, gamma=gamma)

        self.arcface = ArcFace(in_features=out_features, out_features=num_classes)

    def training_step(self, batch, batch_idx):
        rgb, depth, label = batch
        feat_rgb, feat_depth = self.rgbd_fr(rgb, depth)
        logits_rgb = self.arcface(feat_rgb, label)
        logits_depth = self.arcface(feat_depth, label)

        l_cls_rgb = self.ce_rgb(logits_rgb, label)
        l_cls_depth = self.ce_depth(logits_depth, label)

        pred_prob_rgb = softmax(logits_rgb, dim=1).max(dim=1).values
        pred_prob_depth = softmax(logits_depth, dim=1).max(dim=1).values

        l_sa_rgb = self.sa_rgb(feat_rgb, feat_depth, l_cls_rgb, l_cls_depth)
        l_sa_depth = self.sa_depth(feat_depth, feat_rgb, l_cls_depth, l_cls_rgb)

        l_cmfl_rgb = self.cmfl_rgb(pred_prob_rgb, pred_prob_depth)
        l_cmfl_depth = self.cmfl_depth(pred_prob_depth, pred_prob_rgb)

        l_rgb = (1 - self.lambda_1) * l_cls_rgb + \
            self.lambda_1 * l_cmfl_rgb + \
            self.lambda_2 * l_sa_rgb

        l_depth = (1 - self.lambda_2) * l_cls_depth + \
            self.lambda_1 * l_cmfl_depth + \
            self.lambda_2 * l_sa_depth

        return (l_rgb + l_depth).mean()

    def configure_optimizers(self):
        opt = optim.SGD(params=self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        return opt


class RgbdFr(Module):
    def __init__(self, out_features):
        super(RgbdFr, self).__init__()
        self.resnet_rgb = get_branch(out_features=out_features)
        self.resnet_depth = get_branch(out_features=out_features)

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
        feat_rgb = self.resnet_rgb(rgb)
        feat_depth = self.resnet_depth(depth)
        return feat_rgb, feat_depth


def get_pred_cls(logits: torch.Tensor):
    """
    get predicted class
    Args:
        logits: network output logits

    Returns:
        pred: predicted class
    """
    pred = logits.argmax(dim=1)
    return pred


def get_branch(out_features):
    """
    use resnet as backbone
    Args:
        out_features: output feature size

    Returns:
        res: resnet model, the last layer is replaced
    """
    res = resnet18()
    res.fc = nn.Linear(res.fc.in_features, out_features)
    return res


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
        rho = torch.exp(self.beta * (l_cls_m - l_cls_n)) - 1 if l_cls_m > l_cls_n else 0
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
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
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
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output
