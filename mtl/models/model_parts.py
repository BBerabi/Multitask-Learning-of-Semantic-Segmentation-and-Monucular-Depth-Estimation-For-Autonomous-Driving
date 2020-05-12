import torch
import torch.nn.functional as F
import torchvision.models.resnet as resnet


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

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
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
            model = resnet._resnet(
                name, BasicBlockWithDilation, _basic_block_layers[name], pretrained, progress, **encoder_kwargs
            )
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

        # # TODO: Implement a proper decoder with skip connections instead of the following
        # self.features_to_predictions = torch.nn.Conv2d(bottleneck_ch, num_out_ch, kernel_size=1, stride=1)
        #
        self.conv1x1_low_level_features = torch.nn.Sequential(
            torch.nn.Conv2d(skip_4x_ch, skip_4x_ch, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            torch.nn.BatchNorm2d(skip_4x_ch),
            torch.nn.ReLU()
        )
        self.conv3x3_final1 = torch.nn.Sequential(
            torch.nn.Conv2d(skip_4x_ch+bottleneck_ch, bottleneck_ch, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            torch.nn.BatchNorm2d(bottleneck_ch),
            torch.nn.ReLU()
        )
        self.conv3x3_final2 = torch.nn.Conv2d(bottleneck_ch, num_out_ch, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)


    def forward(self, features_bottleneck, features_skip_4x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # print('shape of features bottleneck {}, shape of features_skip_4x {}'.format(features_bottleneck.shape, features_skip_4x.shape))

        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.
        # features_4x = F.interpolate(
        #     features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
        # )
        # # print('shape of features_4x {}'.format(features_4x.shape))
        # predictions_4x = self.features_to_predictions(features_4x)
        # return predictions_4x, features_4x

        ### features_bottleneck are the ones from ASSP
        ### features_skip_4x are the ones from low level encoder

        # upsmaple the bottleneck by 4 or bring it to the same spatial reso as low level
        features_bottleneck_4x = F.interpolate(
             features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
        )
        print('shape of bottleneck features after 4x upsampling: {}'.format(features_bottleneck_4x.shape))
        # apply 1x1 conv to low level features
        features_skip_4x = self.conv1x1_low_level_features(features_skip_4x)
        print('shape of features skip 4x after first 1x1 conv layer {}'.format(features_skip_4x.shape))
        all_features = torch.cat((features_bottleneck_4x, features_skip_4x), 1)
        print('shape of all features {}'.format(all_features.shape))
        ### Apply few conv3x3 layers
        all_features = self.conv3x3_final1(all_features)
        print('shape of all features after forst conv {}'.format(all_features.shape))
        all_features = self.conv3x3_final2(all_features)
        print('shape of all features after second conv {}'.format(all_features.shape))
        return all_features, features_bottleneck_4x


class ASPPpart(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )


class ASPP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, rates=(3, 6, 9)):
        super().__init__()
        print('in channels ASSP: {} out channels ASSP {}'.format(in_channels, out_channels))
        # TODO: Implement ASPP properly instead of the following
        # self.conv_out = ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)

        self.conv1x1 = ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv3x3_rate3 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[0], dilation=rates[0])
        self.conv3x3_rate6 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[1], dilation=rates[1])
        self.conv3x3_rate9 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[2], dilation=rates[2])
        self.conv_out = ASPPpart(256*5, 256, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv1x1_global = ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=1, dilation=1)

    def forward(self, x):
        out1 = self.conv1x1(x)
        out2 = self.conv3x3_rate3(x)
        out3 = self.conv3x3_rate6(x)
        out4 = self.conv3x3_rate9(x)
        # print("Shape of x is {}".format(x.shape))
        # print("Shape of out1 is {}".format(out1.shape))
        # print("Shape of out2 is {}".format(out2.shape))
        # print("Shape of out3 is {}".format(out3.shape))
        # print("Shape of out4 is {}".format(out4.shape))

        ### Global Pooling + 1x1 CNN with 256 filter and BN + upsmaple bilinearly
        self.global_average_pooling = torch.nn.AvgPool2d(x.shape[2:])
        out5 = self.global_average_pooling(x)
        # print("size of out 5 after global average pooling {}".format(out5.shape))
        #now apply 1x1CNN with BN and RELU
        out5 = self.conv1x1_global(out5)
        # print("size of out 5 after conv1x1 {}".format(out5.shape))
        #upsample
        out5 = F.interpolate(
            out5, size=out1.shape[2:], mode='bilinear', align_corners=False
        )
        # print("size of out 5 after upsample {}".format(out5.shape))
        all_features = torch.cat((out1, out2, out3, out4, out5), 1)
        # print("size of all features {}".format(all_features.shape))
        out = self.conv_out(all_features)
        # print("size of out before returning {}".format(out.shape))
        return out


        # TODO: Implement ASPP properly instead of the following
        # out = self.conv_out(x)
        # print('shape of out in initial case {}'.format(out.shape))
        # return out





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
