import copy
import math
import warnings
from functools import partial
from typing import Sequence, Union, Optional, Any, Callable, List

import torch
from torch import nn, Tensor
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.efficientnet import _efficientnet_conf, FusedMBConvConfig, MBConvConfig, \
    _MBConvConfig
from torchvision.ops import Conv2dNormActivation
from torchvision.utils import _log_api_usage_once


class EfficientNet(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
            dropout: float,
            stochastic_depth_prob: float = 0.2,
            num_classes: int = 1000,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            last_channel: Optional[int] = None,
            **kwargs: Any,
    ) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()
        _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
                isinstance(inverted_residual_setting, Sequence)
                and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if "block" in kwargs:
            warnings.warn(
                "The parameter 'block' is deprecated since 0.13 and will be removed 0.15. "
                "Please pass this information on 'MBConvConfig.block' instead."
            )
            if kwargs["block"] is not None:
                for s in inverted_residual_setting:
                    if isinstance(s, MBConvConfig):
                        s.block = kwargs["block"]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                6, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                activation_layer=nn.LeakyReLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, 1),
        )
        self.head = nn.Linear(2048, 16)
        self.head2 = nn.Linear(4096, 512)
        self.head3 = nn.Linear(512, 1)
        self.out = nn.PReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x3 = self.features(x)
        x4 = self.avgpool(x3)
        x5 = torch.flatten(x4, 1)
        x6 = self.head(x5)
        return x6

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _efficientnet(
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        last_channel: Optional[int],
        weights: Optional[WeightsEnum],
        progress: bool,
        **kwargs: Any,
) -> EfficientNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = EfficientNet(inverted_residual_setting, dropout, last_channel=last_channel, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def efficientnet_b7(**kwargs: Any) -> EfficientNet:
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b7", width_mult=2.0, depth_mult=3.1)
    return _efficientnet(
        inverted_residual_setting,
        0.5,
        last_channel,
        None,
        True,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_b5(progress: bool = True, **kwargs: Any
                    ) -> EfficientNet:
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b5", width_mult=1.6, depth_mult=2.2)
    return _efficientnet(
        inverted_residual_setting,
        0.4,
        last_channel,
        None,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


if __name__ == '__main__':
    pass
