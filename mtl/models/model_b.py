import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')
from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p, \
    SelfAttention, DecoderModelB, DecoderModelbType1


class ModelB(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        print('outputs_desc {}'.format(outputs_desc))
        self.outputs_desc = outputs_desc
        ch_out = sum(outputs_desc.values())

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=True,  ########CHANGE TO TRUE WHEN UPLOADING
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )
        print('USING DISTILLATION MODEL :DDDDDDDDDDDDDDD')

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)
        # ch_out_encoder_4x = 512
        print('ch out encoder 4x: ', ch_out_encoder_4x)
        self.aspp_semseg = ASPP(ch_out_encoder_bottleneck, 256)
        self.decoder_semseg = DecoderModelbType1(256, ch_out_encoder_4x, ch_out - 1)

        self.assp_depth = ASPP(ch_out_encoder_bottleneck, 256)
        self.decoder_depth = DecoderModelbType1(256, ch_out_encoder_4x, 1)

        self.sa_from_depth2semseg = SelfAttention(256, 256)
        self.sa_from_semseg2depth = SelfAttention(256, 256)

        self.decoder_semseg2 = DecoderModelB(256, 64, ch_out - 1)
        self.decoder_depth2 = DecoderModelB(256, 64, 1, depth=True)

    def forward(self, x):
        print('shape of x at the beginning: ', x.shape)
        input_resolution = (x.shape[2], x.shape[3])
        # Encoder
        features = self.encoder(x)
        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        print('scales of feature pyramid with their respective number of channels')
        print(", ".join([f"{k}:{v.shape}" for k, v in features.items()]))
        lowest_scale = max(features.keys())
        features_lowest = features[lowest_scale]

        # ASSP Semseg
        features_tasks_semseg = self.aspp_semseg(features_lowest)
        # Decoder Semseg
        features_before_final_layer_semseg = self.decoder_semseg(features_tasks_semseg, features[4])

        # ASSP Depth
        features_tasks_depth = self.assp_depth(features_lowest)
        # Decoder Depth
        features_before_final_layer_depth = self.decoder_depth(features_tasks_depth, features[4])

        attention_from_semseg2depth = self.sa_from_semseg2depth(features_before_final_layer_semseg)
        attention_from_depth2semseg = self.sa_from_depth2semseg(features_before_final_layer_depth)

        final_predictions_semseg = self.decoder_semseg2(features_before_final_layer_semseg, features[2], attention_from_depth2semseg)
        final_predictions_depth = self.decoder_depth2(features_before_final_layer_depth, features[2], attention_from_semseg2depth)


        final_predictions = torch.cat((final_predictions_semseg, final_predictions_depth), 1)

        del final_predictions_semseg
        del final_predictions_depth

        out = {}
        offset = 0
        for task, num_ch in self.outputs_desc.items():
            out[task] = [final_predictions[:, offset:offset + num_ch, :, :]]
            offset += num_ch
        out.stoppp
        return out
