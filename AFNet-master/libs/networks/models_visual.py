

from libs.networks.AFNet_bk import AFNet_backbone
# from libs.modules.eye_fixation_guided import EyeFixationCell, ResGroupBlock
# from adaptive_conv import AdaptiveConv2d
from libs.modules.adaptive_context_filtering import ACFM
from libs.modules.PositionAffinity import TPAM
from libs.modules.temporal_affinity_modeling import TAM

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, groups=1, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ImageModel(nn.Module):
    '''
        RCRNet
    '''
    def __init__(self, cfg, pretrained=False):
        super(ImageModel, self).__init__()
        self.backbone = AFNet_backbone(
            cfg=cfg,
            output_stride=32,
            pretrained=pretrained
        )

    def forward(self, frame):
        seg = self.backbone(frame)
        return seg

    def freeze_bn(self):
        for m in self.backbone.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()


# class TAMHead(nn.Module):
#     def __init__(self, in_channels, add=True):
#         super(TAMHead, self).__init__()
#
#         self.ta_last = TAM(inplanes=in_channels, planes=in_channels//8)
#         # self.sa_self = TAM(inplanes=in_channels, planes=in_channels)
#         self.ta_future = TAM(inplanes=in_channels, planes=in_channels//8)
#         self.ta_fire = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(in_channels)
#         )
#         self.add = add
#
#     def forward(self, last_frame, current_frame, future_frame):
#
#         last_output = self.ta_last(last_frame, current_frame)
#
#         future_output = self.ta_future(future_frame, current_frame)
#
#         ta_output = torch.cat((last_output, future_output), dim=1)
#         ta_output = self.ta_fire(ta_output)
#
#         # tsa_output = self.sa_self(ta_output, ta_output)
#
#         # pa_output = self.pa_fire(feat_add)
#         if self.add:
#             ta_output = F.relu(current_frame + ta_output)
#
#         return ta_output


class GateWeightGenerator(nn.Module):

    def __init__(self, in_channels, num_experts):
        super(GateWeightGenerator, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x)
        # x = self.dropout(x)
        x = self.fc(x)
        return x


class ACFMRear(nn.Module):
    def __init__(self, channel=128, add=True, k1=3, k2=3, k3=3, d1=1, d2=3, d3=5):
        super(ACFMRear, self).__init__()
        self.MDK_front = ACFM(channel=channel//2, k1=k1, k2=k2, k3=k3, d1=d1, d2=d2, d3=d3)
        self.MDK_rear = ACFM(channel=channel//2, k1=k1, k2=k2, k3=k3, d1=d1, d2=d2, d3=d3)
        # self.bn = nn.BatchNorm2d(channel)
        self.add = add
        self.conva = nn.Conv2d(channel, channel//2, 1, padding=0, bias=False)
        self.convc = nn.Conv2d(channel, channel//2, 1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(channel//2, channel, 1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(channel//2, channel, 1, padding=0, bias=False)

        self.Alpha = GateWeightGenerator(channel, 1)
        # self.Alpha2 = GateWeightGenerator(channel, 1)
        # self.Alpha3 = GateWeightGenerator(channel, 1)
        # self.Alpha4 = GateWeightGenerator(channel, 1)

        self.MDK_fire = nn.Sequential(
            nn.Conv2d(in_channels=channel * 2, out_channels=channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel)
        )

    def forward(self, feats_encoder_front, feats_encode, feats_encoder_rear):
        # f_k2_front = feats_encode
        feats_encode1 = self.conva(feats_encode)
        f_k2_front = feats_encode1
        y_front = self.MDK_front(feats_encoder_front, feats_encode1)
        o_k2_front = y_front
        y_front = self.conv1(y_front)
        # o_k2_front = y_front

        feats_encode2 = self.convc(feats_encode)
        f_k2_rear = feats_encode2
        y_rear = self.MDK_rear(feats_encoder_rear, feats_encode2)
        o_k2_rear = y_rear
        y_rear = self.conv2(y_rear)
        # o_k2_rear = y_rear

        dynamic_output = self.MDK_fire(torch.cat((y_front, y_rear), dim=1))
        if self.add:
            alpha = self.Alpha(dynamic_output)
            dynamic_output = alpha * dynamic_output + (1-alpha) * feats_encode
        # concat之后直接用bn
        # dynamic_output = self.bn(dynamic_cat)
        # y = self.MDK_fire(torch.cat((y_front, y_rear), dim=1))

        return dynamic_output, f_k2_front, o_k2_front, f_k2_rear, o_k2_rear


class VideoModel(nn.Module):
    '''
        RCRNet+NER
    '''

    def __init__(self, output_stride=16, pretrained=True, cfg=None):
        super(VideoModel, self).__init__()
        # video mode + video dataset
        self.backbone = AFNet_backbone(
            cfg=cfg,
            output_stride=output_stride,
            pretrained=pretrained
        )

        # self.head_block1 = TAMHead(in_channels=256)
        # self.head_block2 = TAM(in_dim=512)
        # self.head_block3 = TAM(in_dim=1024)
        # self.head_block4 = TAM(in_dim=256)

        self.MDK_module_R3 = ACFMRear(channel=128, k1=3, k2=3, k3=3, d1=1, d2=3, d3=5)
        self.MDK_module_R2 = ACFMRear(channel=128, k1=3, k2=3, k3=3, d1=1, d2=3, d3=5)
        self.MDK_module_R1 = ACFMRear(channel=128, k1=3, k2=3, k3=3, d1=1, d2=3, d3=5)
        self.MDK_module_R0 = ACFMRear(channel=128, k1=3, k2=3, k3=3, d1=1, d2=3, d3=5)

        # self.fuse_heads = self.fuse_head_block()

        # self.freeze_bn()
        if pretrained:
            for key in self.state_dict():
                if 'backbone' not in key:
                    self.video_init_layer(key)
        else:
            for key in self.state_dict():
                self.video_init_layer(key)

    # def fuse_head_block(self):
    #     fuse_heads = [self.head_block1, self.head_block2, self.head_block3, self.head_block4]
    #     return nn.ModuleList(fuse_heads)

    def video_init_layer(self, key):
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                if self.state_dict()[key].ndimension() >= 2:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode='fan_out', nonlinearity='relu')
            elif 'bn' in key:
                self.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            self.state_dict()[key][...] = 0.001

    def freeze_bn(self):
        for m in self.backbone.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()

    def forward(self, clip):

        clip_feats = [self.backbone.feat_conv(frame) for frame in clip]

        # 在上面对block4进行改变了，所以stage4应该拿到这里
        y_list = [self.backbone.stage4(clip_feats[p]) for p in range(4)]

        premask_block1 = []
        premask_block2 = []
        premask_block3 = []
        premask_block4 = []
        f_k2_front_visual = []
        o_k2_front_visual = []
        f_k2_rear_visual = []
        o_k2_rear_visual = []

        # ################# Dynamic convolution 得到1,2,3,4帧的masks  ###  block3处 ######################################
        i = 0
        while i < 4:
            if i == 0:
                feats_encoder_front = y_list[0][3]
                feats_input = y_list[0][3]
                feats_encoder_rear = y_list[1][3]

            elif i == 1:
                feats_encoder_front = y_list[0][3]
                feats_input = y_list[1][3]
                feats_encoder_rear = y_list[2][3]
                premask_block4.append(saliency_feat_res)

            elif i == 2:
                feats_encoder_front = y_list[1][3]
                feats_input = y_list[2][3]
                feats_encoder_rear = y_list[3][3]
                premask_block4.append(saliency_feat_res)

            elif i == 3:
                feats_encoder_front = y_list[2][3]
                feats_input = y_list[3][3]
                feats_encoder_rear = y_list[3][3]
                premask_block4.append(saliency_feat_res)

            saliency_feat_res, f_k2_front_res, o_k2_front_res, f_k2_rear_res, o_k2_rear_res = self.MDK_module_R0(feats_encoder_front, feats_input, feats_encoder_rear)
            i = i + 1
        premask_block4.append(saliency_feat_res)  # premask_block4包含第0到第3帧的block4

        # y_list[k][2]代表第k帧的block3， y_list[k][3]代表第k帧的block4
        feats_encode_block3 = [self.backbone.DenseDecoder.seg_conv(y_list[k][2], premask_block4[k])
                               for k in range(4)]
        # feats_encode_block3代表第0到3帧的bu1
        # ################# Dynamic convolution 得到1,2,3,4帧的masks  ###  block3处 ######################################
        i = 0
        while i < 4:
            if i == 0:
                feats_encoder_front = feats_encode_block3[0]
                feats_input = feats_encode_block3[0]
                feats_encoder_rear = feats_encode_block3[1]

            elif i == 1:
                feats_encoder_front = feats_encode_block3[0]
                feats_input = feats_encode_block3[1]
                feats_encoder_rear = feats_encode_block3[2]
                premask_block3.append(saliency_feat_res)

            elif i == 2:
                feats_encoder_front = feats_encode_block3[1]
                feats_input = feats_encode_block3[2]
                feats_encoder_rear = feats_encode_block3[3]
                premask_block3.append(saliency_feat_res)

            elif i == 3:
                feats_encoder_front = feats_encode_block3[2]
                feats_input = feats_encode_block3[3]
                feats_encoder_rear = feats_encode_block3[3]
                premask_block3.append(saliency_feat_res)

            saliency_feat_res, f_k2_front_res, o_k2_front_res, f_k2_rear_res, o_k2_rear_res = self.MDK_module_R1(feats_encoder_front, feats_input, feats_encoder_rear)
            i = i + 1
        premask_block3.append(saliency_feat_res)    # premask_block3包含第0到第3帧的bu1

        feats_encode_block2 = [self.backbone.DenseDecoder.seg_conv2(y_list[k][1], premask_block4[k], premask_block3[k])
                               for k in range(4)]
        # ################# Dynamic convolution 得到1,2,3,4帧的masks  ###  block2处 ######################################
        i = 0
        while i < 4:
            if i == 0:
                feats_encoder_front = feats_encode_block2[0]
                feats_input = feats_encode_block2[0]
                feats_encoder_rear = feats_encode_block2[1]

            elif i == 1:
                feats_encoder_front = feats_encode_block2[0]
                feats_input = feats_encode_block2[1]
                feats_encoder_rear = feats_encode_block2[2]
                premask_block2.append(saliency_feat_res)

            elif i == 2:
                feats_encoder_front = feats_encode_block2[1]
                feats_input = feats_encode_block2[2]
                feats_encoder_rear = feats_encode_block2[3]
                premask_block2.append(saliency_feat_res)

            elif i == 3:
                feats_encoder_front = feats_encode_block2[2]
                feats_input = feats_encode_block2[3]
                feats_encoder_rear = feats_encode_block2[3]
                premask_block2.append(saliency_feat_res)

            saliency_feat_res, f_k2_front_res, o_k2_front_res, f_k2_rear_res, o_k2_rear_res = self.MDK_module_R2(feats_encoder_front, feats_input, feats_encoder_rear)
            i = i + 1
        premask_block2.append(saliency_feat_res)    # premask_block2包含第0到第3帧的bu2

        feats_encode_block1s = [
            self.backbone.DenseDecoder.seg_conv3(y_list[k][0], premask_block4[k], premask_block3[k], premask_block2[k])
            for k in range(4)]      # 包含第0帧的bu3, block4_1, bu2_1, bu1_1，  第1帧的bu3, block4_1, bu2_1, bu1_1 ...
        # ################# Dynamic convolution 得到1,2,3,4帧的masks  ###  block1处 ######################################
        i = 0
        while i < 4:
            if i == 0:
                feats_encoder_front = feats_encode_block1s[0][0]
                feats_input = feats_encode_block1s[0][0]
                feats_encoder_rear = feats_encode_block1s[1][0]

            elif i == 1:
                feats_encoder_front = feats_encode_block1s[0][0]
                feats_input = feats_encode_block1s[1][0]
                feats_encoder_rear = feats_encode_block1s[2][0]
                premask_block1.append(saliency_feat_res)
                f_k2_front_visual.append(F.sigmoid(f_k2_front_res))
                o_k2_front_visual.append(F.sigmoid(o_k2_front_res))
                f_k2_rear_visual.append(F.sigmoid(f_k2_rear_res))
                o_k2_rear_visual.append(F.sigmoid(o_k2_rear_res))

            elif i == 2:
                feats_encoder_front = feats_encode_block1s[1][0]
                feats_input = feats_encode_block1s[2][0]
                feats_encoder_rear = feats_encode_block1s[3][0]
                premask_block1.append(saliency_feat_res)
                f_k2_front_visual.append(F.sigmoid(f_k2_front_res))
                o_k2_front_visual.append(F.sigmoid(o_k2_front_res))
                f_k2_rear_visual.append(F.sigmoid(f_k2_rear_res))
                o_k2_rear_visual.append(F.sigmoid(o_k2_rear_res))

            elif i == 3:
                feats_encoder_front = feats_encode_block1s[2][0]
                feats_input = feats_encode_block1s[3][0]
                feats_encoder_rear = feats_encode_block1s[3][0]
                premask_block1.append(saliency_feat_res)
                f_k2_front_visual.append(F.sigmoid(f_k2_front_res))
                o_k2_front_visual.append(F.sigmoid(o_k2_front_res))
                f_k2_rear_visual.append(F.sigmoid(f_k2_rear_res))
                o_k2_rear_visual.append(F.sigmoid(o_k2_rear_res))

            saliency_feat_res, f_k2_front_res, o_k2_front_res, f_k2_rear_res, o_k2_rear_res = self.MDK_module_R3(feats_encoder_front, feats_input, feats_encoder_rear)
            i = i + 1
        premask_block1.append(saliency_feat_res)    # premask_block1包含第0到第3帧的bu3
        f_k2_front_visual.append(F.sigmoid(f_k2_front_res))
        o_k2_front_visual.append(F.sigmoid(o_k2_front_res))
        f_k2_rear_visual.append(F.sigmoid(f_k2_rear_res))
        o_k2_rear_visual.append(F.sigmoid(o_k2_rear_res))

        preds = []
        for i, frame in enumerate(clip):
            # y_list = self.backbone.stage4(xlist[i])
            # seg = self.backbone.DenseDecoder(clip_feats[i][0], clip_feats[i][1], clip_feats[i][2], Premask[i], frame)
            # bu2 = self.backbone.DenseDecoder.seg_conv2(y_list[i][1], y_list[i][3], premask[i])
            # seg = self.backbone.DenseDecoder(y_list[0], y_list[1], y_list[2], y_list[3], frame)
            # side_sups = self.backbone.DenseDecoder.side_sup(y_list[i][3], premask_block3[i], premask_block2[i], premask_block1[i])
            seg = self.backbone.DenseDecoder.segment(premask_block1[i], feats_encode_block1s[i][1],
                                                     feats_encode_block1s[i][2], feats_encode_block1s[i][3],
                                                     frame.shape[2:])
            # block4_s.append(side_sups[0])
            # bu1_s.append(side_sups[1])
            # bu2_s.append(side_sups[2])
            # bu3_s.append(side_sups[3])
            preds.append(torch.sigmoid(seg))
        return preds, f_k2_front_visual, o_k2_front_visual, f_k2_rear_visual, o_k2_rear_visual



