import math
import torch
from torch import nn, cosine_similarity, Tensor
from torch.nn import Module
from torch.nn.functional import linear, normalize
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models.resnet import BasicBlock, ResNet
from model.senet import SENet


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

        return rho * (0.5 - cos) ** 2


def backbone_factory(backbone="resnet18", *args, **kwargs):
    """
    use resnet as backbone
    Args:
        backbone: backbone network

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


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(
            2, 1, kernel_size, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class RecognitionBranch(ResNet):
    def __init__(self, out_features, block=BasicBlock, layers=(2, 2, 2, 2)):
        super(RecognitionBranch, self).__init__(
            block=block,
            layers=list(layers)
        )
        self.fc = nn.Linear(self.fc.in_features, out_features)

    def forward(self, x, attention1=1, attention3=1):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x * attention1)
        x = self.layer3(x)
        x = self.layer4(x * attention3)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class AuxiliaryBranch(ResNet):
    def __init__(self, block=BasicBlock, layers=(2, 2, 2, 2)):
        super(AuxiliaryBranch, self).__init__(
            block=block,
            layers=list(layers)
        )
        self.sam = SpatialAttention()
        del self.fc, self.avgpool, self.layer4

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x1 = self.sam(x1)
        x3 = self.sam(x3)

        return x1, x3


class RgbdFr(Module):
    def __init__(self, out_features):
        super(RgbdFr, self).__init__()
        self.rgb_net = RecognitionBranch(out_features)
        self.depth_net = RecognitionBranch(out_features)
        self.segment_net = AuxiliaryBranch()
        # self.attention = {}

        # self.segment_net.layer1.register_forward_hook(
        #     self._get_intermediate_feature_map(1)
        # )

    # def _get_intermediate_feature_map(self, layer):
    #     def hook(module, x, y):
    #         self.attention[layer] = y
    #     return hook

    # def spatial_attention(self):
    #     # attention shape: (x, x, 1)
    #     attention_1 = self.sam(self.attention[1])
    #     attention_3 = self.sam(self.attention[3])
    #     return attention_1, attention_3

    def forward(self, rgb, depth, segment=None):
        """
        get rgb and depth branches' feature vectors
        Args:
            rgb: rgb image
            depth: depth image
            segment: segment image

        Returns:
            feat_rgb: rgb feature vector
            feat_depth: depth feature vector
        """
        if segment is not None:
            # self.attention[3] = self.segment_net(segment)
            # attention[1]: (56, 56, 64), attention[3]: (14, 14, 256)
            attention_1, attention_3 = self.segment_net(segment)
            # attention[1]: (56, 56, 1), attention[3]: (14, 14, 1)
        else:
            attention_1, attention_3 = (1, 1)

        feat_rgb = self.rgb_net(rgb, attention_1, attention_3)
        feat_depth = self.depth_net(depth, attention_1, attention_3)
        return feat_rgb, feat_depth
