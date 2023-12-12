import torch
import torch.nn as nn
import math

from torch.nn import functional as F



__all__ = ['scale_casa']
class ChannelAttention_HAM(nn.Module):
    def __init__(self, in_channel):
        super(ChannelAttention_HAM, self).__init__()

        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.alpha.data.fill_(0.5)
        self.beta.data.fill_(0.5)

        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.b = 1
        self.y = 2

        self.k = math.log(in_channel, 2) / self.y + self.b / self.y

        if int(self.k) % 2 == 0:
            self.K = int(self.k) + 1
        else:
            self.K = int(self.k)


        self.conv1d = nn.Conv1d(1, 1, kernel_size=self.K,padding=self.K//2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #plan A
        #print("x size ", x.size())
        '''
        maxout = self.maxpool(x).view(x.size(0), -1)
        avgout = self.avgpool(x).view(x.size(0), -1)

        Fc = 0.5*(maxout+avgout)+self.alpha*avgout+self.beta*maxout
        #
        Fc = Fc.unsqueeze(1)
        #print("fc size ", Fc.size())
        #pri

        Fc = self.conv1d(Fc)
        Fc = Fc.squeeze(1)
        Fc = Fc.unsqueeze(2).unsqueeze(3)
        #print("fc size ", Fc.size())
        #pri



        #plan B
        '''
        avgout = self.avgpool(x)
        maxout = self.maxpool(x)
        Fc = 0.5 * (maxout + avgout) + self.alpha * avgout + self.beta * maxout
        Fc = Fc.squeeze(-1).permute(0, 2, 1)
        Fc = self.conv1d(Fc).permute(0, 2, 1).unsqueeze(-1)




        return self.sigmoid(Fc)

class SpatialAttention_HAM(nn.Module):
    def __init__(self, in_channel, separation_rate):
        super(SpatialAttention_HAM,self).__init__()

        self.separation_rate = separation_rate
        self.in_channel = in_channel

        self.h = self.separation_rate * self.in_channel

        if int(self.h) % 2 == 0:
            self.C_im = int(self.h)
        else:
            self.C_im = int(self.h) + 1

        self.C_subim = self.in_channel - self.C_im

        #kernel_size = 1
        #padding = 3 if kernel_size == 7 else 0

        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

        #self.conv1 = nn.Conv2d(in_channel, in_channel // 2, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.nam = NAM(in_channel)
        self.relu = nn.ReLU()
        #self.bn2 = nn.BatchNorm2d(1, affine=True)
        self.bn2 = nn.BatchNorm2d(1)

        #self.im_mask = torch.ones(batch_size, in_channel,input_hight * input_width).cuda()
        #self.subim_mask = torch.ones(batch_size, in_channel, input_hight * input_width).cuda()

    def get_im_subim_channel(self, C_im, M):
        _, topk = torch.topk(M, dim=1, k = C_im)
        important_channel = torch.zeros_like(M)
        subimportant_channel = torch.ones_like(M)
        important_channel = important_channel.scatter(1, topk, 1)
        subimportant_channel = subimportant_channel.scatter(1, topk, 0)
        return important_channel, subimportant_channel

    def get_feature(self,im_channel, subim_channel, channel_refined_feature):
        import_feature = im_channel*channel_refined_feature
        subimportant_feature = subim_channel*channel_refined_feature
        return import_feature,subimportant_feature

    def forward(self, x, M):
        im_mask, subim_mask = self.get_im_subim_channel(self.C_im, M)
        im_features, subim_features = self.get_feature(im_mask, subim_mask, x)
        '''
        subim_features = x * subim_mask
        # x = self.nam(x)
        #print("x size", x.size())
        x_sepa = x.view(x.size(0),x.size(1),-1)
        x_sepa = torch.mean(x_sepa, dim=-1, keepdim=True)
        a, idx = torch.sort(x_sepa, 1, descending=True)
        im_filed = a[:, :self.C_im, :]
        #print("im_filed_size ", im_filed.size())
        #im_filed_idx = idx[:, :self.C_im, :]
        #print("a size", a.size())
        #print("a ", a)
        #print("idx size", idx.size())
        #print("idx", idx)

        nth_im = im_filed[:, self.C_im - 1, :]
        #print("nth_im size", nth_im.size())
        #im_filed, _ = torch.min(im_filed,dim=-1,keepdim=True)
        #print("im_filed ", im_filed)
        #pri
        #B,C,L = x.size()
        #im_mask = torch.ones(x.size(0), x.size(1),x.size(2) * x.size(3)).cuda()
        #subim_mask = torch.ones(x.size(0), x.size(1), x.size(2) * x.size(3)).cuda()
        #print("x_sepa[i,j,:] ",x_sepa[1, 1, :].item())
        #print("nth_im[i,:,:] ", nth_im[1, :].item())
        #pri
        for i in range(6):
            for j in range(self.in_channel):
                if x_sepa[i,j,:].item() >= nth_im[i,:].item():
                    self.subim_mask[i, j, :] = 0
                else:
                    self.im_mask[i,j,:] = 0

        subim_mask = self.subim_mask.view(x.size(0),x.size(1),x.size(2), x.size(3))
        im_mask = self.im_mask.view(x.size(0),x.size(1),x.size(2), x.size(3))
        #print("subim_mask", subim_mask.size())
        #print("im_mask", im_mask.size())
        #pri

        #im_mask = im_mask.view(x.size(0),x.size(1),x.size(2) * x.size(3))
        #subim_mask = subim_mask.view(x.size(0), x.size(1), x.size(2) * x.size(3))



        im_features = x* im_mask
        subim_features = x* subim_mask
        '''


        avgout_im = torch.mean(im_features, dim=1, keepdim=True)*(self.in_channel/self.C_im)
        maxout_im, _ = torch.max(im_features, dim=1, keepdim=True)
        #print("avgout_im size", avgout_im.size())
        #print("maxout_im size", maxout_im.size())

        im = self.sigmoid(self.relu(self.bn2(self.conv(torch.cat([avgout_im, maxout_im], dim=1)))))

        avgout_subim = torch.mean(subim_features, dim=1, keepdim=True)*(self.in_channel/self.C_subim)
        maxout_subim, _ = torch.max(subim_features, dim=1, keepdim=True)
        #out_subim = self.sigmoid(self.conv(torch.cat([maxout_subim, avgout_subim], dim=1)))
        subim = self.sigmoid(self.relu(self.bn2(self.conv(torch.cat([avgout_subim, maxout_subim], dim=1)))))

        out_im = im*im_features
        out_subim = subim*subim_features


        #x = self.conv1(x)
        #avgout = torch.mean(x, dim=1, keepdim=True)
        #maxout, _ = torch.max(x, dim=1, keepdim=True)
        #out = self.sigmoid(self.conv(torch.cat([maxout, avgout], dim=1)))
        return out_im, out_subim

class DS_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DS_Conv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.depth_conv = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, padding=1, groups=self.in_channel)
        self.point_conv = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class CASA_HAM(nn.Module):
    def __init__(self, in_channel, separation_rate=0.6):
        super(CASA_HAM,self).__init__()

        #self.bn2 = nn.BatchNorm2d(in_channel, affine=False)


        self.CA = ChannelAttention_HAM(in_channel)
        self.SA = SpatialAttention_HAM(in_channel, separation_rate)
        self.relu = nn.ReLU()


        #self.sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU()

        #self.conv = torch.nn.Sequential(
            #nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.ReLU(),
            #nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self,x):

        #residual = x
        #x = self.conv(x)
        #x = self.sigmoid(0.8*(1.0/self.sigmoid(x)-0.2*x))
        #x = -x
        #x = self.sigmoid(-x)
        #x = self.sigmoid(x.exp())
        #x = self.sigmoid(1-0.8*x)
        #x = self.sigmoid((1.0 / (self.sigmoid(x) + 0.1)))
        #x = self.sigmoid(1.0 / (self.sigmoid(x)+0.00001))
        #x = self.sigmoid(1.0 / (self.sigmoid(x) + 0.1))
        #x = self.sigmoid(1.0 / (self.sigmoid(x)))
        #x = self.bn2(x)
        #x = self.sigmoid(-x)


        #out = self.CA(x)*x
        channel_att_map = self.CA(x)
        channel_refine_feature = channel_att_map * x
        #out = (1-0.8*self.SA(out))*out
        out_im, out_subim = self.SA(channel_refine_feature, channel_att_map)
        output = out_im+out_subim

        return self.relu(output+x)

class scale_casa_HAM(nn.Module):
    def __init__(self, num_channel):
        super(scale_casa_HAM, self).__init__()

        '''
        self.casa_96 = CASA(in_channel=num_channel[0])
        self.casa_192 = CASA(in_channel=num_channel[1])
        self.casa_384 = CASA(in_channel=num_channel[2])
        self.casa_768 = CASA(in_channel=num_channel[3])
        '''


        self.casa_96 = CASA_HAM(in_channel=num_channel[0])
        self.casa_192 = CASA_HAM(in_channel=num_channel[1])
        self.casa_384 = CASA_HAM(in_channel=num_channel[2])
        self.casa_768 = CASA_HAM(in_channel=num_channel[3])

        #self.Down_ch = torch.nn.Sequential(nn.Conv2d(sum(num_channel), 8, kernel_size=3, stride=1, padding=1, bias=False),
                                           #nn.ELU())
        self.Down_ch = torch.nn.Sequential(DS_Conv(sum(num_channel), 64),
                                           nn.ELU())
        '''
        self.Down_ch = torch.nn.Sequential(
            nn.Conv2d(sum(num_channel), 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU())
        '''


        self.elu = nn.ELU()

    def forward(self, feature):
        features = []

        f0 = feature[0]
        f0 = self.casa_96(f0)
        features.append(f0)
        #print('f0 size ', f0.size())
        f0 = F.interpolate(f0, scale_factor=2, mode='nearest')
        #print('f0 size ',f0.size())


        f1 = feature[1]
        f1 = self.casa_192(f1)
        features.append(f1)
        #print('f1 size ', f1.size())
        f1 = F.interpolate(f1, scale_factor=4, mode='nearest')
        #print('f1 size ', f1.size())


        f2 = feature[2]
        f2 = self.casa_384(f2)
        features.append(f2)
        #print('f2 size ', f2.size())
        f2 = F.interpolate(f2, scale_factor=8, mode='nearest')
        #print('f2 size ', f2.size())


        #feature 3 [320,6,20]
        '''
        f3 = feature[3]
        f3 = self.casa_768(f3)
        #features.append(f3)
        #print('f3 size ', f3.size())
        f3 = F.interpolate(f3, scale_factor=16, mode='nearest')
        #print('f3 size ', f3.size())
        '''


        # feature 4 [320,6,20]
        f3_4 = feature[4]
        f3_4 = self.casa_768(f3_4)
        features.append(f3_4)
        # print('f3 size ', f3.size())
        f3_4 = F.interpolate(f3_4, scale_factor=16, mode='nearest')
        # print('f3 size ', f3.size())


        f4 = torch.cat([f0,f1,f2,f3_4], dim=1)
        f4 = self.Down_ch(f4)
        #print('f4 size ', f4.size())
        features.insert(0, f4)

        '''
        print('features length ', len(features))
        print('features 0 size ', features[0].size())
        print('features 1 size ', features[1].size())
        print('features 2 size ', features[2].size())
        print('features 3 size ', features[3].size())
        print('features 4 size ', features[4].size())
        pri
        '''




        return features









