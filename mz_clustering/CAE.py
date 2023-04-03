import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import math
import torchvision.transforms as transforms


def conv2d_hout(height, padding, dilation, kernel_size, stride):
    tmp = math.floor(((height + 2*padding[0] - dilation[0]*(kernel_size[0] - 1) - 1) / stride[0]) + 1)
    return 1 if tmp < 1 else tmp


def conv2d_wout(width, padding, dilation, kernel_size, stride):
    tmp = math.floor(((width + 2*padding[1] - dilation[1]*(kernel_size[1] - 1) - 1) / stride[1]) + 1)
    return 1 if tmp < 1 else tmp


class CAE(nn.Module):
    def __init__(self, height, width, train_mode=True,):
        super(CAE, self).__init__()
        self.train_mode = train_mode
        # self.fc_h1, self.fc_h2 = 768, 256
        self.encoder_dim = 7
        self.k1, self.k2 = (3, 3), (3, 3)
        self.s1, self.s2 = (2, 2), (3, 3)
        self.height = height
        self.width = width
        self.d1, self.d2 = 8, 16

        # encoder
        self.conv1 = nn.Sequential(nn.Conv2d(1, self.d1, kernel_size=self.k1, stride=self.s1, padding=(0, 0), dilation=(1, 1),
                                             bias=False),
                                   nn.BatchNorm2d(self.d1, momentum=0.01),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.d1, self.d2, kernel_size=self.k2, stride=self.s2, padding=(0, 0), bias=False),
                                   nn.BatchNorm2d(self.d2, momentum=0.01),
                                   nn.ReLU())

        # Calculating dimensions of encoding
        self.l1height = conv2d_hout(height=height, padding=(0, 0), dilation=(1, 1),
                                    kernel_size=self.k1, stride=self.s1)
        self.l1width = conv2d_wout(width=width, padding=(0, 0), dilation=(1, 1),
                                   kernel_size=self.k1, stride=self.s1)
        self.l2height = conv2d_hout(height=self.l1height, padding=(0, 0), dilation=(1, 1),
                                    kernel_size=self.k2, stride=self.s2)
        self.l2width = conv2d_wout(width=self.l1width, padding=(0, 0), dilation=(1, 1),
                                   kernel_size=self.k2, stride=self.s2)

        self.encoder = nn.Linear(self.l2height*self.l2width*self.d2, self.encoder_dim)

        # decoder
        self.fc3 = nn.Linear(self.encoder_dim, self.l2height*self.l2width*self.d2)
        self.ct1 = nn.ConvTranspose2d(in_channels=self.d2, out_channels=self.d1, kernel_size=self.k2,
                                      stride=self.s2, padding=(0, 0), dilation=(1, 1),
                                      output_padding=(1, 1))
        self.tbn1 = nn.BatchNorm2d(self.d1, momentum=0.01)
        self.trelu1 = nn.ReLU()
        self.ct2 = nn.ConvTranspose2d(in_channels=self.d1, out_channels=1, kernel_size=self.k1,
                                      stride=self.s1, padding=(0, 0), dilation=(1, 1),
                                      output_padding=(1, 1))
        self.tbn2 = nn.BatchNorm2d(1, momentum=0.01)
        self.trelu2 = nn.ReLU()

        # self.conv_trans1 = nn.Sequential(nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3, 3),
        #                                                     stride=(3, 3), padding=(0, 0), dilation=(1, 1),
        #                                                     output_padding=(1, 1)),
        #                                  nn.BatchNorm2d(8, momentum=0.01),
        #                                  nn.ReLU(inplace=True)
        #                                  )
        #
        # self.conv_trans2 = nn.Sequential(nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(3, 3),
        #                                                     stride=(2, 2), padding=(0, 0), dilation=(1, 1),
        #                                                     output_padding=(1, 1)),
        #                                  nn.BatchNorm2d(1, momentum=0.01),
        #                                  nn.Sigmoid()
        #                                  )

        # self.conv_trans3 = nn.Sequential(nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=self.k3, stride = self.s3, padding = (0,0)),
        #                                  nn.BatchNorm2d(1, momentum = 0.01),
        #                                  nn.Sigmoid()
        #                                  )
    def convtrans1(self, x, output_size):
        x = self.ct1(x, output_size=output_size)
        x = self.tbn1(x)
        x = self.trelu1(x)
        return x

    def convtrans2(self, x, output_size):
        x = self.ct2(x, output_size=output_size)
        x = self.tbn2(x)
        x = self.trelu2(x)
        return x

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        print(x.shape)
        print(self.l2height)
        print(self.l2width)
        print(self.d2)
        print(self.l2height*self.l2width*self.d2)
        x = self.encoder(x)
        return x

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(-1, 16, self.l2height, self.l2width)
        z = self.convtrans1(z, output_size=(self.l1height, self.l1width))
        z = self.convtrans2(z, output_size=(self.height, self.width))
        z = F.interpolate(z, size=(self.height, self.width), mode='bilinear')
        return z

    def forward(self, x):
        # mu, sigma = self.encode(x)
        z = self.encode(x)
        xp = self.decode(z)
        return xp
