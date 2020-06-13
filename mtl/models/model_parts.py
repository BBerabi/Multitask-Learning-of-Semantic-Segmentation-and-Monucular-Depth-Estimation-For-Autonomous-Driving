import torch
import torch.nn.functional as F
import torchvision.models.resnet as resnet
from torch.hub import load_state_dict_from_url

# from torchvision.utils import load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class BasicBlockWithDilation(torch.nn.Module):
    """Workaround for prohibited dilation in BasicBlock in 0.4.0"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockWithDilation, self).__init__()

        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = resnet.conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = resnet.conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = SqueezeAndExcitation(planes)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        # add S&E mechanism
        out = self.se(out)

        out += identity
        out = self.relu(out)
        return out


_basic_block_layers = {
    'resnet18': (2, 2, 2, 2),
    'resnet34': (3, 4, 6, 3),
}


def get_encoder_channel_counts(encoder_name):
    is_basic_block = encoder_name in _basic_block_layers
    ch_out_encoder_bottleneck = 512 if is_basic_block else 2048
    ch_out_encoder_4x = 64 if is_basic_block else 256
    return ch_out_encoder_bottleneck, ch_out_encoder_4x


class Encoder(torch.nn.Module):
    def __init__(self, name, **encoder_kwargs):
        super().__init__()
        encoder = self._create(name, **encoder_kwargs)
        del encoder.avgpool
        del encoder.fc
        self.encoder = encoder

    def _create(self, name, **encoder_kwargs):
        if name not in _basic_block_layers.keys():
            fn_name = getattr(resnet, name)
            model = fn_name(**encoder_kwargs)
        else:
            # special case due to prohibited dilation in the original BasicBlock
            pretrained = encoder_kwargs.pop('pretrained', False)
            progress = encoder_kwargs.pop('progress', True)
            model = resnet.ResNet(BasicBlockWithDilation, _basic_block_layers[name], **encoder_kwargs)
            if pretrained:
                state_dict = load_state_dict_from_url(model_urls[name])
                model.load_state_dict(state_dict, strict=False)
            # model = resnet._resnet(
            #     name, BasicBlockWithDilation, _basic_block_layers[name], pretrained, progress, **encoder_kwargs
            # )
        replace_stride_with_dilation = encoder_kwargs.get('replace_stride_with_dilation', (False, False, False))
        assert len(replace_stride_with_dilation) == 3
        if replace_stride_with_dilation[0]:
            model.layer2[0].conv2.padding = (2, 2)
            model.layer2[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[1]:
            model.layer3[0].conv2.padding = (2, 2)
            model.layer3[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[2]:
            model.layer4[0].conv2.padding = (2, 2)
            model.layer4[0].conv2.dilation = (2, 2)
        return model

    def update_skip_dict(self, skips, x, sz_in):
        rem, scale = sz_in % x.shape[3], sz_in // x.shape[3]
        assert rem == 0
        skips[scale] = x

    def forward(self, x):
        """
        DeepLabV3+ style encoder
        :param x: RGB input of reference scale (1x)
        :return: dict(int->Tensor) feature pyramid mapping downscale factor to a tensor of features
        """
        out = {1: x}
        sz_in = x.shape[3]

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.maxpool(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer1(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer2(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer3(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer4(x)
        self.update_skip_dict(out, x, sz_in)

        return out


class DecoderDeeplabV3p(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_4x_ch, num_out_ch):
        super(DecoderDeeplabV3p, self).__init__()
        print('number of out channels in decoderdeeplabv3p {}'.format(num_out_ch))

        print('number of skip4x channels in decoderdeeplabv3p {}'.format(skip_4x_ch))
        print('number of bottleneck channels in decoderdeeplabv3p {}'.format(bottleneck_ch))

        number_lowlevel = 48
        self.conv1x1_low_level_features = torch.nn.Sequential(
            torch.nn.Conv2d(skip_4x_ch, number_lowlevel, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            torch.nn.BatchNorm2d(number_lowlevel),
            torch.nn.ReLU()
        )
        self.conv3x3_final1 = torch.nn.Sequential(
            torch.nn.Conv2d(number_lowlevel + bottleneck_ch, bottleneck_ch, kernel_size=3, stride=1, padding=1,
                            dilation=1, bias=False),
            torch.nn.BatchNorm2d(bottleneck_ch),
            torch.nn.ReLU()
        )
        self.conv3x3_final2 = torch.nn.Sequential(
            torch.nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, stride=1, padding=1, dilation=1,
                            bias=False),
            torch.nn.BatchNorm2d(bottleneck_ch),
            torch.nn.ReLU()
        )

        self.conv3x3_final3 = torch.nn.Conv2d(bottleneck_ch, num_out_ch, kernel_size=3, stride=1, padding=1, dilation=1,
                                              bias=False)

    def forward(self, features_bottleneck, features_skip_4x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        features_bottleneck_4x = F.interpolate(
            features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
        )
        # apply 1x1 conv to low level features
        features_skip_4x = self.conv1x1_low_level_features(features_skip_4x)
        all_features = torch.cat((features_bottleneck_4x, features_skip_4x), 1)
        # apply few conv3x3 layers
        all_features = self.conv3x3_final1(all_features)
        distillation_features = self.conv3x3_final2(all_features)
        output = self.conv3x3_final3(distillation_features)
        return output, distillation_features


class ASPPpart(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, separable=False,
                 depthwise_multiplier=1):
        if not separable:
            super().__init__(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
            )
        else:
            super().__init__(
                torch.nn.Conv2d(in_channels, in_channels * depthwise_multiplier, kernel_size, stride, padding, dilation,
                                bias=False, groups=in_channels),
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                dilation=dilation, bias=False),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
            )


class ASPP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, rates=(6, 12, 18)):
        super().__init__()

        self.conv1x1 = ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv3x3_rate3 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[0],
                                      dilation=rates[0], separable=True)
        self.conv3x3_rate6 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[1],
                                      dilation=rates[1], separable=True)
        self.conv3x3_rate9 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[2],
                                      dilation=rates[2], separable=True)
        self.conv_out = ASPPpart(256 * 5, 256, kernel_size=1, stride=1, padding=0, dilation=1)

        self.global_average_pooling = torch.nn.AvgPool2d((37, 49))  # just set to some kernel size, later update
        self.conv1x1_global = ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=1, dilation=1)

    def forward(self, x):
        out1 = self.conv1x1(x)
        out2 = self.conv3x3_rate3(x)
        out3 = self.conv3x3_rate6(x)
        out4 = self.conv3x3_rate9(x)

        # Global Pooling + 1x1 CNN with 256 filter and BN + upsmaple bilinearly
        self.global_average_pooling.kernel_size = x.shape[2:]  # update the kernel size
        out5 = self.global_average_pooling(x)

        # now apply 1x1CNN with BN and RELU
        out5 = self.conv1x1_global(out5)
        # upsample
        out5 = F.interpolate(
            out5, size=out1.shape[2:], mode='bilinear', align_corners=False
        )

        all_features = torch.cat((out1, out2, out3, out4, out5), 1)
        out = self.conv_out(all_features)
        return out


class SelfAttention(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.attention = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.attention.weight.copy_(torch.zeros_like(self.attention.weight))

    def forward(self, x):
        features = self.conv(x)
        attention_mask = torch.sigmoid(self.attention(x))
        return features * attention_mask


class SqueezeAndExcitation(torch.nn.Module):
    """
    Squeeze and excitation module as explained in https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, channels, r=16):
        super(SqueezeAndExcitation, self).__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // r),
            torch.nn.ReLU(),
            torch.nn.Linear(channels // r, channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        N, C, H, W = x.shape
        squeezed = torch.mean(x, dim=(2, 3)).reshape(N, C)
        squeezed = self.transform(squeezed).reshape(N, C, 1, 1)
        return x * squeezed


class DecoderDistillation(torch.nn.Module):
    def __init__(self, in_channels, num_out_ch):
        super(DecoderDistillation, self).__init__()

        self.conv3x3_final1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1, dilation=1,
                            bias=False),
            torch.nn.BatchNorm2d(in_channels//2),
            torch.nn.ReLU()
        )

        self.conv3x3_final2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, stride=1, padding=1, dilation=1,
                            bias=False),
            torch.nn.BatchNorm2d(in_channels//4),
            torch.nn.ReLU()
        )

        self.conv3x3_final3 = torch.nn.Conv2d(in_channels//4, num_out_ch, kernel_size=3, stride=1, padding=1, dilation=1,
                                              bias=False)

    def forward(self, features_low, features_sa):
        """
        """
        all_features = features_low + features_sa
        all_features = self.conv3x3_final1(all_features)
        target_shape = (all_features.shape[2]*2, all_features.shape[3]*2)
        all_features = F.interpolate(
            all_features, size=target_shape, mode='bilinear', align_corners=False
        )

        all_features = self.conv3x3_final2(all_features)
        target_shape = (all_features.shape[2] * 2, all_features.shape[3] * 2)
        all_features = F.interpolate(
            all_features, size=target_shape, mode='bilinear', align_corners=False
        )

        all_features = self.conv3x3_final3(all_features)
        return all_features


class DecoderTransposed(torch.nn.Module):
    def __init__(self, in_channels, num_out_ch):
        super(DecoderTransposed, self).__init__()

        # self.se1 = SqueezeAndExcitation(in_channels)

        self.transconv3x3_final1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm2d(in_channels // 2),
            torch.nn.ReLU()
        )

        # self.se2 = SqueezeAndExcitation(in_channels // 2)

        self.transconv3x3_final2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=3, stride=2, padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm2d(in_channels // 4),
            torch.nn.ReLU()
        )

        # self.se3 = SqueezeAndExcitation(in_channels // 4)

        self.conv3x3_final3 = torch.nn.Conv2d(in_channels // 4, num_out_ch, kernel_size=3, stride=1, padding=1,
                                              dilation=1,
                                              bias=False)

    def forward(self, features_low, features_sa):
        all_features = features_low + features_sa
        # all_features = self.se1(all_features)
        all_features = self.transconv3x3_final1(all_features)
        # all_features = self.se2(all_features)
        all_features = self.transconv3x3_final2(all_features)
        # all_features = self.se3(all_features)
        all_features = self.conv3x3_final3(all_features)
        return all_features


class DecoderUnet(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_8x_ch, num_out_ch):
        super(DecoderUnet, self).__init__()
        skip_4x_ch = skip_8x_ch // 2
        skip_2x_ch = skip_4x_ch
        number_low_level = 48
        self.conv1x1_low_level_8xfeatures = torch.nn.Sequential(
            torch.nn.Conv2d(skip_8x_ch, number_low_level, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU()
        )

        self.conv1x1_low_level_4xfeatures = torch.nn.Sequential(
            torch.nn.Conv2d(skip_4x_ch, number_low_level, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU()
        )

        self.conv1x1_low_level_2xfeatures = torch.nn.Sequential(
            torch.nn.Conv2d(skip_2x_ch, number_low_level, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU()
        )

        self.conv3x3_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(bottleneck_ch + number_low_level, 256, kernel_size=3, stride=1, padding=1, dilation=1,
                            bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        self.up8x4x = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 180, kernel_size=3, stride=2, padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm2d(180),
            torch.nn.ReLU()
        )

        self.conv3x3_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(228, 160, kernel_size=3, stride=1, padding=1, dilation=1,
                            bias=False),
            torch.nn.BatchNorm2d(160),
            torch.nn.ReLU()
        )

        self.up4x2x = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(160, 120, kernel_size=3, stride=2, padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm2d(120),
            torch.nn.ReLU()
        )

        self.conv3x3_block3 = torch.nn.Sequential(
            torch.nn.Conv2d(168, 128, kernel_size=3, stride=1, padding=1, dilation=1,
                            bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )

        bias = True if num_out_ch == 1 else False
        self.conv3x3_final = torch.nn.Conv2d(128, num_out_ch, kernel_size=3, stride=1, padding=1, dilation=1,
                                             bias=bias)

    def forward(self, features, features_skip_8x, features_skip_4x, features_skip_2x):
        features = F.interpolate(
            features, size=features_skip_8x.shape[2:], mode='bilinear', align_corners=False
        )
        features_skip_8x = self.conv1x1_low_level_8xfeatures(features_skip_8x)
        features = torch.cat((features, features_skip_8x), 1)
        features = self.conv3x3_block1(features)

        features = self.up8x4x(features)
        features_skip_4x = self.conv1x1_low_level_4xfeatures(features_skip_4x)
        features = torch.cat((features, features_skip_4x), 1)
        features = self.conv3x3_block2(features)

        features = self.up4x2x(features)
        features_skip_2x = self.conv1x1_low_level_2xfeatures(features_skip_2x)
        features = torch.cat((features, features_skip_2x), 1)
        features_distillation = self.conv3x3_block3(features)

        features = self.conv3x3_final(features_distillation)
        return features, features_distillation


class DecoderSkip(torch.nn.Module):
    def __init__(self, in_channels, skip_2x_ch, num_out_ch):
        super(DecoderSkip, self).__init__()

        number_lowlevel = 32
        self.conv1x1_low_level_features = torch.nn.Sequential(
            torch.nn.Conv2d(skip_2x_ch, number_lowlevel, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            torch.nn.BatchNorm2d(number_lowlevel),
            torch.nn.ReLU()
        )

        self.transconv3x3_final1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm2d(in_channels // 2),
            torch.nn.ReLU()
        )

        self.conv3x3_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels // 2 + + number_lowlevel, in_channels // 2, kernel_size=3, stride=1, padding=1,
                            dilation=1,
                            bias=False),
            torch.nn.BatchNorm2d(in_channels // 2),
            torch.nn.ReLU()
        )

        self.conv3x3_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1, dilation=1,
                            bias=False),
            torch.nn.BatchNorm2d(in_channels // 2),
            torch.nn.ReLU()
        )

        self.transconv3x3_final2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=3, stride=2, padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm2d(in_channels // 4),
            torch.nn.ReLU()
        )

        self.conv3x3_block3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1, dilation=1,
                            bias=False),
            torch.nn.BatchNorm2d(in_channels // 4),
            torch.nn.ReLU()
        )

        self.conv3x3_block4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1, dilation=1,
                            bias=False),
            torch.nn.BatchNorm2d(in_channels // 4),
            torch.nn.ReLU()
        )

        self.conv3x3_final = torch.nn.Conv2d(in_channels // 4, num_out_ch, kernel_size=3, stride=1, padding=1,
                                             dilation=1,
                                             bias=False)

    def forward(self, features_low, features_sa, features_skip_2x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        all_features = features_low + features_sa
        all_features = self.transconv3x3_final1(all_features)
        features_skip_2x = self.conv1x1_low_level_features(features_skip_2x)
        all_features = torch.cat((all_features, features_skip_2x), 1)

        all_features = self.conv3x3_block1(all_features)
        all_features = self.conv3x3_block2(all_features)
        all_features = self.transconv3x3_final2(all_features)

        all_features = self.conv3x3_block3(all_features)
        all_features = self.conv3x3_block4(all_features)
        all_features = self.conv3x3_final(all_features)
        return all_features


class BilinearAdditiveResidualUpsampling(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BilinearAdditiveResidualUpsampling, self).__init__()
        self.N = in_channels // out_channels  # this has to be exactly divisable
        self.conv3x3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels // self.N, in_channels // self.N, kernel_size=3, stride=1, padding=1, dilation=1,
                            bias=False),
            torch.nn.BatchNorm2d(in_channels // self.N),
            torch.nn.ReLU()
        )

    def forward(self, x):
        # it always upsample by a factor of 2
        target_shape = (x.shape[2] * 2, x.shape[3] * 2)
        x = F.interpolate(x, size=target_shape, mode='bilinear', align_corners=False)
        # now apply channel reduction by averaging
        # sum every self.N consecutive channels
        n, c, h, w = x.shape
        identity = torch.reshape(x, (n, c // self.N, self.N, h, w))
        identity = torch.sum(identity, dim=2)

        x = self.conv3x3(identity)
        return x + identity


class DecoderModelB(torch.nn.Module):
    def __init__(self, in_channels, skip_ch_2x, num_out_ch, depth=False):
        super(DecoderModelB, self).__init__()
        self.up1 = BilinearAdditiveResidualUpsampling(in_channels, in_channels // 4)
        number_lowlevel = 16
        self.conv1x1_2x = torch.nn.Sequential(
            torch.nn.Conv2d(skip_ch_2x, number_lowlevel, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            torch.nn.BatchNorm2d(number_lowlevel),
            torch.nn.ReLU()
        )

        self.conv3x3_final1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels // 4 + number_lowlevel, in_channels // 4 + number_lowlevel, kernel_size=3,
                            stride=1, padding=1, dilation=1, bias=False),
            torch.nn.BatchNorm2d(in_channels // 4 + number_lowlevel),
            torch.nn.ReLU()
        )
        last_channels = (in_channels // 4 + number_lowlevel) // 4
        self.up2 = BilinearAdditiveResidualUpsampling(in_channels // 4 + number_lowlevel, last_channels)
        self.conv3x3_final2 = torch.nn.Sequential(
            torch.nn.Conv2d(last_channels, last_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                            bias=False),
            torch.nn.BatchNorm2d(last_channels),
            torch.nn.ReLU()
        )
        self.conv3x3_final3 = torch.nn.Conv2d(last_channels, num_out_ch, kernel_size=3, stride=1, padding=1,
                                              dilation=1,
                                              bias=False)

    def forward(self, x, features_skip2x, features_sa):
        x = x + features_sa
        x = self.up1(x)
        features_skip2x = self.conv1x1_2x(features_skip2x)
        x = torch.cat((x, features_skip2x), 1)
        x = self.conv3x3_final1(x)
        x = self.up2(x)
        x = self.conv3x3_final2(x)
        x = self.conv3x3_final3(x)
        return x

class DecoderModelbType1(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_4x_ch, num_out_ch):
        super(DecoderModelbType1, self).__init__()

        number_lowlevel = 48
        self.conv1x1_low_level_features = torch.nn.Sequential(
            torch.nn.Conv2d(skip_4x_ch, number_lowlevel, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            torch.nn.BatchNorm2d(number_lowlevel),
            torch.nn.ReLU()
        )
        self.conv3x3_final1 = torch.nn.Sequential(
            torch.nn.Conv2d(number_lowlevel + bottleneck_ch, bottleneck_ch, kernel_size=3, stride=1, padding=1,
                            dilation=1, bias=False),
            torch.nn.BatchNorm2d(bottleneck_ch),
            torch.nn.ReLU()
        )
        self.conv3x3_final2 = torch.nn.Sequential(
            torch.nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, stride=1, padding=1, dilation=1,
                            bias=False),
            torch.nn.BatchNorm2d(bottleneck_ch),
            torch.nn.ReLU()
        )


    def forward(self, features_bottleneck, features_skip_4x):
        # features_bottleneck are the ones from ASSP
        # features_skip_4x are the ones from low level encoder

        # upsmaple the bottleneck by 4 or bring it to the same spatial reso as low level
        features_bottleneck_4x = F.interpolate(
            features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
        )
        # apply 1x1 conv to low level features
        features_skip_4x = self.conv1x1_low_level_features(features_skip_4x)
        all_features = torch.cat((features_bottleneck_4x, features_skip_4x), 1)
        # apply few conv3x3 layers
        all_features = self.conv3x3_final1(all_features)
        all_features = self.conv3x3_final2(all_features)
        return all_features

