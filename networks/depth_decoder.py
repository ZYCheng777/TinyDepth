# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layer import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([64, 18, 36, 72, 144])# [16, 32, 64, 128, 256] [64, 18, 36, 72, 144]

        #self.CASA = scale_casa([128, 256, 384, 512])
        #self.CASA = scale_casa([18, 36, 72, 144])
        #self.CASA = scale_casa_HAM_T([18, 36, 72, 144])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        #self.Down_ch = ConvBlock(64, 32)

    def forward(self, input_feature):
        self.outputs = {}
        #input_feature = self.CASA(input_features)

        # decoder
        '''
        print('input size !!!', input_feature[4].size())
        print('input size !!!', input_feature[3].size())
        print('input size !!!', input_feature[2].size())
        print('input size !!!', input_feature[1].size())
        print('input size !!!', input_feature[0].size())

        input_features = []
        # out = input_feature[0]
        e = upsample(input_feature[0])
        # out = e
        e = self.Down_ch(e)
        input_features.append(e)
        input_features.append(input_feature[0])
        input_features.append(input_feature[1])
        input_features.append(input_feature[2])
        input_features.append(input_feature[4])

        print('input size !!!', input_features[4].size())
        print('input size !!!', input_features[3].size())
        print('input size !!!', input_features[2].size())
        print('input size !!!', input_features[1].size())
        print('input size !!!', input_features[0].size())
        '''


        x = input_feature[-1]
        for i in range(4, -1, -1):
            #print('x size !!!',x.size())
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            #print('x size', x[0].size())
            if self.use_skips and i > 0:
                x += [input_feature[i - 1]]
                #print('x+ size !!!', x[0].size(),x[1].size())
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        #pri

        return self.outputs
