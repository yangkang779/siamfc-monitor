import numpy as np

import scipy.io
import sys
import six
import os.path
from PIL import Image, ImageStat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from src.crops import extract_crops_z, extract_crops_x, pad_frame
import cv2
from torch.nn import DataParallel
sys.path.append('../')

class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x**2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq,pad,pad,pad,pad),2),
                            torch.cat((pad,x_sq,pad,pad,pad),2),
                            torch.cat((pad,pad,x_sq,pad,pad),2),
                            torch.cat((pad,pad,pad,x_sq,pad),2),
                            torch.cat((pad,pad,pad,pad,x_sq),2)),1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:,2:-2,:,:]
        x = x / ((2.+0.0001*x_sumsq)**0.75)
        return x

class SiameseNet(nn.Module):
    def __init__(self, root_pretrained=None, net=None):
        super(SiameseNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 32, 3, 1, groups=2),
        )
        self.mid_feature = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
        )
        self.branch = nn.Sequential(
            # self.conv1,
            # self.conv2,
            # self.conv3,
            self.conv4,
            self.conv5
        )
        # make feature map channels dont change, and size (H*W) same the last layer size
        # conv4 output:[1, 384, 12, 12]--conv5--[1, 32, 6, 6]--transconv--[1, 32, 12, 12]

        # self.trans_conv3 = nn.ConvTranspose2d(384, 384, 12, stride=1, bias=False)
        # self.trans_conv3.weight.data = self.bilinear_kernel(384, 384, 12)
        # self.trans_conv5 = nn.ConvTranspose2d(32, 32, 16, stride=1, bias=False)
        # self.trans_conv5.weight.data = self.bilinear_kernel(32, 32, 16)

        # self.con3_pool = nn.MaxPool2d(5, 1)

        self.bn_adjust = nn.BatchNorm2d(1)
        self.lrn = LRN()
        # self._initialize_weights()
        # self.cuda = torch.cuda.is_available()
        if net is not None:
            net_path = os.path.join(root_pretrained, net)
            load_siamfc_from_matconvnet(net_path, self)
            for m in self.modules():
                m.training = False

    def forward(self, x):

        m = self.mid_feature(x)
        x = self.branch(m)
        # m = self.con3_pool(m)
        # m = self.lrn(m)
        # m = torch.chunk(m, 16, 1)
        # x = torch.cat([m[0], x], 1)
        # x = self.lrn(x)

        return x

    def xcorr(self, z, x):
        out = F.conv2d(x, z)
        return out

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()

    # def bilinear_kernel(self, in_channels, out_channels, kernel_size):
    #     factor = (kernel_size + 1) // 2  # // return integer part of result
    #     if kernel_size % 2 == 1:
    #         center = factor - 1
    #     else:
    #         center = factor - 0.5
    #     og = np.ogrid[:kernel_size, :kernel_size]
    #     filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    #     weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    #     weight[range(in_channels), range(out_channels), :, :] = filt
    #     return torch.from_numpy(weight)

    def get_template_z(self, pos_x, pos_y, z_sz, image, design):
        if isinstance(image, six.string_types):
            image = Image.open(image)
        avg_chan = ImageStat.Stat(image).mean
        frame_padded_z, npad_z = pad_frame(image, image.size, pos_x, pos_y, z_sz, avg_chan)
        z_crops = extract_crops_z(frame_padded_z, npad_z, pos_x, pos_y, z_sz, design.exemplar_sz)

        template_z = self.forward(Variable(z_crops)) # 1*32*17*17
        return image, template_z

    def get_scores(self, pos_x, pos_y, scaled_search_area, template_z, filename, design, final_score_sz):
        image = Image.open(filename)
        avg_chan = ImageStat.Stat(image).mean
        frame_padded_x, npad_x = pad_frame(image, image.size, pos_x, pos_y, scaled_search_area[2], avg_chan)
        x_crops = extract_crops_x(frame_padded_x, npad_x, pos_x, pos_y, scaled_search_area[0], scaled_search_area[1], scaled_search_area[2], design.search_sz)

        template_x = self.forward(Variable(x_crops)) # 3*32*49*49
        scores = self.xcorr(template_z, template_x)
        scores = self.bn_adjust(scores) # 3*1*33*33
        # TODO: any elegant alternator?
        scores = scores.squeeze().permute(1, 2, 0).cpu().data.numpy() # 33*33*3
        scores_up = cv2.resize(scores, (final_score_sz, final_score_sz))
        scores_up = scores_up.transpose((2, 0, 1))
        return image, scores_up


def load_siamfc_from_matconvnet(net_path, model):
    params_names_list, params_values_list = load_matconvnet(net_path)

    params_values_list = [torch.from_numpy(p) for p in params_values_list]
    for l, p in enumerate(params_values_list):
        param_name = params_names_list[l]
        if 'conv' in param_name and param_name[-1] == 'f':
            p = p.permute(3, 2, 0, 1)
        p = torch.squeeze(p)
        params_values_list[l] = p

    net = nn.Sequential(
        model.conv1,
        model.conv2,
        model.conv3,
        model.conv4,
        model.conv5
    )

    for l, layer in enumerate(net):
        layer[0].weight.data[:] = params_values_list[params_names_list.index('br_conv%df' % (l + 1))]
        layer[0].bias.data[:] = params_values_list[params_names_list.index('br_conv%db' % (l + 1))]

        if l < len(net) - 1:
            layer[1].weight.data[:] = params_values_list[params_names_list.index('br_bn%dm' % (l + 1))]
            layer[1].bias.data[:] = params_values_list[params_names_list.index('br_bn%db' % (l + 1))]

            bn_moments = params_values_list[params_names_list.index('br_bn%dx' % (l + 1))]
            layer[1].running_mean[:] = bn_moments[:,0]
            layer[1].running_var[:] = (bn_moments[:,1] ** 2)
        else:
            model.bn_adjust.weight.data[:] = params_values_list[params_names_list.index('fin_adjust_bnm')]
            model.bn_adjust.bias.data[:] = params_values_list[params_names_list.index('fin_adjust_bnb')]

            bn_moments = params_values_list[params_names_list.index('fin_adjust_bnx')].cuda()
            model.bn_adjust.running_mean[:] = bn_moments[0]
            model.bn_adjust.running_var[:] = (bn_moments[1] ** 2)

    return model

def load_matconvnet(net_path):
    mat = scipy.io.loadmat(net_path)
    net_dot_mat = mat.get('net')
    params = net_dot_mat['params']
    params = params[0][0]
    params_names = params['name'][0]
    params_names_list = [params_names[p][0] for p in range(params_names.size)]
    params_values = params['value'][0]
    params_values_list = [params_values[p] for p in range(params_values.size)]

    return params_names_list, params_values_list