import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
# from torch.nn import functional as F


class TPAM(Module):
    """ Temporal Position affinity module"""

    def __init__(self, in_dim):
        super(TPAM, self).__init__()
        self.chanel_in = in_dim

        # if in_dim == 256:
        #     self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        #     self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        # elif in_dim == 128:
        #     self.query_conv = Conv2d(in_channels=in_dim * 2, out_channels=in_dim//2, kernel_size=1)
        #     self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)

        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # gamma从0开始好还是从1开始好
        self.gamma = Parameter(torch.zeros(1))
        # self.gamma = Parameter(torch.ones(1))

        # self.conv_E = Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)

        self.softmax = Softmax(dim=-1)

    def forward(self, last_frame, current_frame):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        # Inter-frame position modeling
        m_batchsize, C, height, width = current_frame.size()
        # proj_query = self.query_conv(last_frame).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        # proj_key = self.key_conv(current_frame).view(m_batchsize, -1, width*height)

        # 翻转上下
        proj_query = self.query_conv(last_frame).view(m_batchsize, -1, width * height)
        proj_key = self.key_conv(current_frame).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        energy = torch.bmm(proj_key, proj_query)
        attention = self.softmax(energy)
        proj_value = self.value_conv(current_frame).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        # In-frame position modeling
        # waiting to do...

        out = self.gamma*out + current_frame
        # gamma = self.gamma
        return out
