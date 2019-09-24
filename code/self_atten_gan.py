import numpy as np

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch.backends.cudnn as cudnn

from layers import init_weights
from layers import snlinear
from layers import snconv2d
from layers import sn_embedding
from layers import GenBlock
from layers import Self_Attn
from layers import DiscOptBlock
from layers import DiscBlock

class Generator(nn.Module):
    """Generator."""

    def __init__(self, z_dim, g_conv_dim, num_classes):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.g_conv_dim = g_conv_dim

        self.snlinear0 = snlinear(in_features=z_dim, out_features=g_conv_dim * 8 * 4 * 4)

        self.block1 = GenBlock(g_conv_dim * 8, g_conv_dim * 8, num_classes)
        self.block2 = GenBlock(g_conv_dim * 8, g_conv_dim * 4, num_classes)
        self.block3 = GenBlock(g_conv_dim * 4, g_conv_dim * 2, num_classes)

        self.self_attn = Self_Attn(g_conv_dim * 2)

        self.block4 = GenBlock(g_conv_dim * 2, g_conv_dim, num_classes)

        self.bn = nn.BatchNorm2d(g_conv_dim, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.snconv2d1 = snconv2d(in_channels=g_conv_dim, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # Weight init
        self.apply(init_weights)

    def forward(self, z, labels):
        # z = self.pixelnorm(z)

        # n x z_dim -> n x g_conv_dim*8*4*4(8192)
        act0 = self.snlinear0(z)

        # n x g_conv_dim*8*4*4(8192) -> n x g_conv_dim*8 x 4 x 4(n x 512 x 4 x 4)
        act0 = act0.view(-1, self.g_conv_dim * 8, 4, 4)

        # n x 512 x 4 x 4 ->  x 512 x 8 x 8
        act1 = self.block1(act0, labels)

        # n x 512 x 8 x 8 -> n x 256 x 16 x 16
        act2 = self.block2(act1, labels)

        # act2 = self.attn1(act2)

        # n x 256 x 16 x 16 -> n x 128 x 32 x 32
        act3 = self.block3(act2, labels)

        # n x 128 x 32 x 32 -> n x 128 x 32 x 32
        act3 = self.self_attn(act3)

        # n x 128 x 32 x 32 -> n x 64 x 64 x 64
        act4 = self.block4(act3, labels)

        # act4 = self.attn2(act4)
        # act5 = self.block5(act4, labels)

        act4 = self.bn(act4)
        act4 = self.relu(act4)

        # n x 64 x 64 x 64 -> n x 3 x 64 x 64
        act5 = self.snconv2d1(act4)

        # act5 = self.pixelnorm(act5)

        #  n x 3 x 64 x 64
        act5 = self.tanh(act5)
        return act5


class Discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, d_conv_dim, num_classes):
        super(Discriminator, self).__init__()
        self.d_conv_dim = d_conv_dim
        self.opt_block1 = DiscOptBlock(3, d_conv_dim)
        self.block1 = DiscBlock(d_conv_dim, d_conv_dim * 2)
        self.self_attn = Self_Attn(d_conv_dim * 2)
        self.block2 = DiscBlock(d_conv_dim * 2, d_conv_dim * 4)
        self.block3 = DiscBlock(d_conv_dim * 4, d_conv_dim * 8)
        self.block5 = DiscBlock(d_conv_dim * 8, d_conv_dim * 8)
        self.relu = nn.ReLU(inplace=True)
        self.snlinear1 = snlinear(in_features=d_conv_dim * 8, out_features=1)
        self.sn_embedding1 = sn_embedding(num_classes, d_conv_dim * 8)

        # Weight init
        self.apply(init_weights)
        xavier_uniform_(self.sn_embedding1.weight)

    def forward(self, x, labels):
        # x = n x 3 x 64 x 64
        # n x 3 x 64 x 64  ->  n x d_conv_dim   x 32 x 32
        h0 = self.opt_block1(x)

        # n x d_conv_dim x 32 x 32  ->  n x d_conv_dim*2 x 16 x 16
        h1 = self.block1(h0)

        # n x d_conv_dim*2 x 16 x 16    n x d_conv_dim*2 x 16 x 16
        h1 = self.self_attn(h1)

        # n x d_conv_dim*2 x 16 x 16 -> n x d_conv_dim*4 x  8 x  8
        h2 = self.block2(h1)

        # n x d_conv_dim*4 x 8 x 8 -> n x d_conv_dim*8 x 4 x 4
        h3 = self.block3(h2)

        # h4 = self.block4(h3)    # n x d_conv_dim*16 x 4 x  4

        # n x d_conv_dim*8 x 4 x 4 -> # n x d_conv_dim*8 x 4 x 4
        h5 = self.block5(h3, downsample=False)
        h5 = self.relu(h5)

        # n x d_conv_dim*8 x 4 x 4 -> n x d_conv_dim*8
        h6 = torch.sum(h5, dim=[2, 3])

        # n x d_conv_dim*8 -> n x 1
        output1 = torch.squeeze(self.snlinear1(h6))

        # Projection  -> n x d_conv_dim*8
        h_labels = self.sn_embedding1(labels)

        # n x d_conv_dim*8 -> n x d_conv_dim*8
        proj = torch.mul(h6, h_labels)

        # n x d_conv_dim*8 -> n x 1
        output2 = torch.sum(proj, dim=[1])

        # Out: n x 1
        output = output1 + output2
        return output

