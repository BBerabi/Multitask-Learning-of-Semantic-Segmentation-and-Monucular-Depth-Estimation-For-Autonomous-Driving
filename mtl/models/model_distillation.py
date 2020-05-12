import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p


class ModelDistillation(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        print('outputs_desc {}'.format(outputs_desc))
        self.outputs_desc = outputs_desc
        ch_out = sum(outputs_desc.values())

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=False, ########CHANGE TO TRUE WHEN UPLOADING
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )
        print('USING DISTILLATION MODEL :DDDDDDDDDDDDDDD')
        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        self.aspp_semseg = ASPP(ch_out_encoder_bottleneck, 256)
        self.decoder_semseg = DecoderDeeplabV3p(256, ch_out_encoder_4x, ch_out-1)

        self.assp_depth = ASPP(ch_out_encoder_bottleneck, 256)
        self.decoder_depth = DecoderDeeplabV3p(256, ch_out_encoder_4x, 1)


    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])
        input_resolution.stoppp
        # Encoder
        features = self.encoder(x)
        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        print('scales of feature pyramid with their respective number of channels')
        print(", ".join([f"{k}:{v.shape}" for k, v in features.items()]))
        lowest_scale = max(features.keys())
        print('lowest scale: {}'.format(lowest_scale))
        features_lowest = features[lowest_scale]

        # ASSP Semseg
        features_tasks_semseg = self.aspp_semseg(features_lowest)
        print("shape of feature_tasks_semseg {} and shape of features[4] {}".format(features_tasks_semseg.shape, features[4].shape))
        # Decoder Semseg
        predictions_4x_semseg, _ = self.decoder_semseg(features_tasks_semseg, features[4])
        print("shape of predictions_4x_semseg(decoder output) {}".format(predictions_4x_semseg.shape))
        predictions_1x_semseg = F.interpolate(predictions_4x_semseg, size=input_resolution, mode='bilinear', align_corners=False)
        print("shape of predictions_1x_semseg {}".format(predictions_1x_semseg.shape))

        # ASSP Depth
        features_tasks_depth = self.assp_depth(features_lowest)
        print("shape of feature_tasks_depth {} and shape of features[4] {}".format(features_tasks_semseg.shape, features[4].shape))

        # Decoder Depth
        predictions_4x_depth, _2 = self.decoder_depth(features_tasks_depth, features[4])
        print("shape of predictions_4x_depth(decoder output) {}".format(predictions_4x_semseg.shape))
        predictions_1x_depth = F.interpolate(predictions_4x_depth, size=input_resolution, mode='bilinear', align_corners=False)
        print("shape of predictions_1x_depth {}".format(predictions_1x_semseg.shape))


        predictions_1x = torch.cat((predictions_1x_semseg, predictions_1x_depth), 1)

        out = {}
        offset = 0
        for task, num_ch in self.outputs_desc.items():
            out[task] = predictions_1x[:, offset:offset+num_ch, :, :]
            offset += num_ch

        return out
