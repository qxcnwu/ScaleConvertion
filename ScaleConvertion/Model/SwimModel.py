# -*- coding: utf-8 -*-
# @Time    : 2023/2/20 9:28
# @Author  : qxcnwu
# @FileName: SwimModel.py
# @Software: PyCharm
from functools import partial
from typing import List, Optional, Callable, Any

import torch
from torch import nn
from torchvision.models._api import WeightsEnum
from torchvision.models.swin_transformer import SwinTransformerBlock, PatchMerging, Swin_V2_S_Weights, \
    SwinTransformerBlockV2, PatchMergingV2, Swin_V2_B_Weights
from torchvision.ops import Permute
from torchvision.utils import _log_api_usage_once


class SwinTransformer(nn.Module):
    """
    Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/pdf/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
            self,
            patch_size: List[int],
            embed_dim: int,
            depths: List[int],
            num_heads: List[int],
            window_size: List[int],
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            stochastic_depth_prob: float = 0.1,
            num_classes: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            block: Optional[Callable[..., nn.Module]] = None,
            downsample_layer: Callable[..., nn.Module] = PatchMerging,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.num_classes = num_classes

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5, elementwise_affine=False)

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        self.q = nn.Conv2d(6, embed_dim, kernel_size=(patch_size[0], patch_size[1]),
                           stride=(patch_size[0], patch_size[1]))
        layers.append(
            nn.Sequential(
                self.q,
                Permute([0, 2, 3, 1]),
                # nn.BatchNorm2d(embed_dim),
                # Permute([0, 2, 3, 1]),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2 ** i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.permute = Permute([0, 3, 2, 1])
        self.avgpool = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(num_features, 16)
        self.head2 = nn.Linear(16, 1)
        # self.out1 = nn.PReLU()
        # self.head3 = nn.Linear(1024, 256)
        # self.head4 = nn.Linear(256, 1)
        # self.out = torch.nn.SiLU()
        nn.init.xavier_normal_(self.q.weight, )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.norm(x1)
        x3 = self.permute(x2)
        x4 = self.avgpool(x3)
        x5 = torch.squeeze(x4)
        x6 = self.head(x5)

        return x6


def _swin_transformer(
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        stochastic_depth_prob: float,
        weights: Optional[WeightsEnum],
        progress: bool,
        **kwargs: Any,
) -> SwinTransformer:
    model = SwinTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def swin_v2_s(*, weights: Optional[Swin_V2_S_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_v2_small architecture from
    `Swin Transformer V2: Scaling Up Capacity and Resolution <https://arxiv.org/pdf/2111.09883>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_V2_S_Weights
        :members:
    """
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 4, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[16, 16],
        stochastic_depth_prob=0.3,
        weights=None,
        progress=progress,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )


def swin_v2_b(*, weights: Optional[Swin_V2_B_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_v2_base architecture from
    `Swin Transformer V2: Scaling Up Capacity and Resolution <https://arxiv.org/pdf/2111.09883>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_V2_B_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_V2_B_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_V2_B_Weights
        :members:
    """
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[24, 24],
        stochastic_depth_prob=0.5,
        weights=None,
        progress=progress,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )


if __name__ == '__main__':
    model = swin_v2_b(num_classes=4)
    from torchsummary import summary
    summary(model, [[6, 128, 128], [48]], device='cpu')
