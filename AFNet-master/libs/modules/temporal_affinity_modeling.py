import torch
import torch.nn as nn
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F


class TAM(Module):
    """ Temporal affinity module (relation, dependency)"""

    def \
            __init__(self, in_dim):
        super(TAM, self).__init__()
        self.chanel_in = in_dim

        self.query_key = Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.key_last = Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_future = Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.ta_fire = nn.Sequential(
            nn.Conv2d(in_channels=in_dim // 2, out_channels=in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim)
        )

        self.softmax = Softmax(dim=-1)

    def forward(self, last_frame, current_frame, future_frame):
        """

        """
        bs, C, height, width = current_frame.size()

        query_key_map = self.query_key(current_frame).view(bs, -1, width * height).permute(0, 2, 1)
        last_key_map = self.key_last(last_frame).view(bs, -1, width * height)
        future_key_map = self.key_future(future_frame).view(bs, -1, width * height)
        value_map = self.value_conv(current_frame).view(bs, -1, width * height)

        affinity_last_cur = torch.matmul(query_key_map, last_key_map)
        affinity_future_cur = torch.matmul(query_key_map, future_key_map)
        affinity_last_cur = self.softmax(affinity_last_cur)
        affinity_future_cur = self.softmax(affinity_future_cur)

        key_query_map = query_key_map.permute(0, 2, 1)

        out_last_cur = torch.matmul(value_map, affinity_last_cur.permute(0, 2, 1))
        out_last_cur = out_last_cur.view(bs, C // 4, height, width)

        out_future_cur = torch.matmul(value_map, affinity_future_cur.permute(0, 2, 1))
        out_future_cur = out_future_cur.view(bs, C // 4, height, width)

        # fusing
        out_affinity = self.ta_fire(torch.cat((out_last_cur, out_future_cur), dim=1))
        out_affinity = F.relu(current_frame + out_affinity)

        return out_affinity
