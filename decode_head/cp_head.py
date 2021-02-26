from mmcv.cnn.bricks.norm import build_norm_layer
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from torch import nn
from torch.nn.modules.utils import _pair
import numpy as np

from mmseg.core import add_prefix
from mmseg.ops import resize
from ..builder import HEADS, build_loss
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .decode_head import BaseDecodeHead


class AggregationModule(nn.Module):
    """Aggregation Module"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 conv_cfg=None,
                 norm_cfg=None):
        super(AggregationModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        padding = kernel_size // 2

        self.reduce_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU'))

        self.t1 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=out_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        self.t2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=out_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)

        self.p1 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=out_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        self.p2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=out_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        _, self.norm = build_norm_layer(norm_cfg, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward function."""
        x = self.reduce_conv(x)
        x1 = self.t1(x)
        x1 = self.t2(x1)

        x2 = self.p1(x)
        x2 = self.p2(x2)

        out = self.relu(self.norm(x1 + x2))
        return out


@HEADS.register_module()
class CPHead(BaseDecodeHead):
    """Context Prior for Scene Segmentation.

    This head is the implementation of `CPNet
    <https://arxiv.org/abs/2004.01547>`_.
    """

    def __init__(self,
                 prior_channels,
                 prior_size,
                 am_kernel_size,
                 groups=1,
                 loss_prior_decode=dict(type='AffinityLoss', loss_weight=1.0),
                 **kwargs):
        super(CPHead, self).__init__(**kwargs)
        self.prior_channels = prior_channels
        self.prior_size = _pair(prior_size)
        self.am_kernel_size = am_kernel_size

        self.aggregation = AggregationModule(self.in_channels, prior_channels,
                                             am_kernel_size, self.conv_cfg,
                                             self.norm_cfg)

        self.prior_conv = ConvModule(
            self.prior_channels,
            np.prod(self.prior_size),
            1,
            padding=0,
            stride=1,
            groups=groups,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.intra_conv = ConvModule(
            self.prior_channels,
            self.prior_channels,
            1,
            padding=0,
            stride=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.inter_conv = ConvModule(
            self.prior_channels,
            self.prior_channels,
            1,
            padding=0,
            stride=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.bottleneck = ConvModule(
            self.in_channels + self.prior_channels * 2,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.loss_prior_decode = build_loss(loss_prior_decode)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        batch_size, channels, height, width = x.size()
        assert self.prior_size[0] == height and self.prior_size[1] == width

        value = self.aggregation(x)

        context_prior_map = self.prior_conv(value)
        context_prior_map = context_prior_map.view(batch_size,
                                                   np.prod(self.prior_size),
                                                   -1)
        context_prior_map = context_prior_map.permute(0, 2, 1)
        context_prior_map = torch.sigmoid(context_prior_map)

        inter_context_prior_map = 1 - context_prior_map

        value = value.view(batch_size, self.prior_channels, -1)
        value = value.permute(0, 2, 1)

        intra_context = torch.bmm(context_prior_map, value)
        intra_context = intra_context.div(np.prod(self.prior_size))
        intra_context = intra_context.permute(0, 2, 1).contiguous()
        intra_context = intra_context.view(batch_size, self.prior_channels,
                                           self.prior_size[0],
                                           self.prior_size[1])
        intra_context = self.intra_conv(intra_context)

        inter_context = torch.bmm(inter_context_prior_map, value)
        inter_context = inter_context.div(np.prod(self.prior_size))
        inter_context = inter_context.permute(0, 2, 1).contiguous()
        inter_context = inter_context.view(batch_size, self.prior_channels,
                                           self.prior_size[0],
                                           self.prior_size[1])
        inter_context = self.inter_conv(inter_context)

        cp_outs = torch.cat([x, intra_context, inter_context], dim=1)
        output = self.bottleneck(cp_outs)
        output = self.cls_seg(output)

        return output, context_prior_map

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``pam_cam`` is used."""
        return self.forward(inputs)[0]

    def _construct_ideal_affinity_matrix(self, label, label_size):
        scaled_labels = F.interpolate(
            label.float(), size=label_size, mode="nearest")
        scaled_labels = scaled_labels.squeeze_().long()
        scaled_labels[scaled_labels == 255] = self.num_classes
        one_hot_labels = F.one_hot(scaled_labels, self.num_classes + 1)
        one_hot_labels = one_hot_labels.view(
            one_hot_labels.size(0), -1, self.num_classes + 1).float()
        ideal_affinity_matrix = torch.bmm(one_hot_labels,
                                          one_hot_labels.permute(0, 2, 1))
        return ideal_affinity_matrix

    def losses(self, seg_logit, seg_label):
        """Compute ``seg``, ``prior_map`` loss."""
        seg_logit, context_prior_map = seg_logit
        logit_size = seg_logit.shape[2:]
        loss = dict()
        loss.update(super(CPHead, self).losses(seg_logit, seg_label))
        prior_loss = self.loss_prior_decode(
            context_prior_map,
            self._construct_ideal_affinity_matrix(seg_label, logit_size))
        loss['loss_prior'] = prior_loss
        return loss
