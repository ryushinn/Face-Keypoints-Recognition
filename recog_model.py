from torch import nn
import torch
import torch.nn.functional as F
from math import ceil, floor


class FaceRecognition(nn.Module):
    def __init__(self):
        super(FaceRecognition, self).__init__()
        self.conv_layer = nn.Sequential()
        self.conv_layer.add_module("Conv1", nn.Conv2d(3, 64, 3, padding=1))
        self.conv_layer.add_module("BN1", nn.BatchNorm2d(64, affine=True))
        self.conv_layer.add_module("Act_layer1", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("Conv2", nn.Conv2d(64, 64, 3, padding=1))
        self.conv_layer.add_module("BN2", nn.BatchNorm2d(64, affine=True))
        self.conv_layer.add_module("Act_layer2", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("Conv11", nn.Conv2d(64, 64, 3, padding=1))
        self.conv_layer.add_module("BN11", nn.BatchNorm2d(64, affine=True))
        self.conv_layer.add_module("Act_layer11", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("MaxPool1", nn.MaxPool2d(2, 2))

        self.conv_layer.add_module("Conv3", nn.Conv2d(64, 128, 3, padding=1))
        self.conv_layer.add_module("BN3", nn.BatchNorm2d(128, affine=True))
        self.conv_layer.add_module("Act_layer3", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("Conv4", nn.Conv2d(128, 128, 3, padding=1))
        self.conv_layer.add_module("BN4", nn.BatchNorm2d(128, affine=True))
        self.conv_layer.add_module("Act_layer4", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("Conv12", nn.Conv2d(128, 128, 3, padding=1))
        self.conv_layer.add_module("BN12", nn.BatchNorm2d(128, affine=True))
        self.conv_layer.add_module("Act_layer12", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("MaxPool2", nn.MaxPool2d(2, 2))

        self.conv_layer.add_module("Conv5", nn.Conv2d(128, 256, 3, padding=1))
        self.conv_layer.add_module("BN5", nn.BatchNorm2d(256, affine=True))
        self.conv_layer.add_module("Act_layer5", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("Conv6", nn.Conv2d(256, 256, 3, padding=1))
        self.conv_layer.add_module("BN6", nn.BatchNorm2d(256, affine=True))
        self.conv_layer.add_module("Act_layer6", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("Conv13", nn.Conv2d(256, 256, 3, padding=1))
        self.conv_layer.add_module("BN13", nn.BatchNorm2d(256, affine=True))
        self.conv_layer.add_module("Act_layer13", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("MaxPool3", nn.MaxPool2d(2, 2))

        self.conv_layer.add_module("Conv7", nn.Conv2d(256, 512, 3, padding=1))
        self.conv_layer.add_module("BN7", nn.BatchNorm2d(512, affine=True))
        self.conv_layer.add_module("Act_layer7", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("Conv8", nn.Conv2d(512, 512, 3, padding=1))
        self.conv_layer.add_module("BN8", nn.BatchNorm2d(512, affine=True))
        self.conv_layer.add_module("Act_layer8", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("Conv14", nn.Conv2d(512, 512, 3, padding=1))
        self.conv_layer.add_module("BN14", nn.BatchNorm2d(512, affine=True))
        self.conv_layer.add_module("Act_layer14", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("MaxPool4", nn.MaxPool2d(2, 2))

        self.conv_layer.add_module("Conv9", nn.Conv2d(512, 1024, 3, padding=1))
        self.conv_layer.add_module("BN9", nn.BatchNorm2d(1024, affine=True))
        self.conv_layer.add_module("Act_layer9", nn.LeakyReLU(0.1))
        self.conv_layer.add_module("Conv10", nn.Conv2d(1024, 1024, 3, padding=1))
        self.conv_layer.add_module("BN10", nn.BatchNorm2d(1024, affine=True))
        self.conv_layer.add_module("Act_layer10", nn.LeakyReLU(0.1))

        # self.spp_layer = SPPLayer(3, "max_pool")
        #
        # self.fc = nn.Sequential()
        # self.fc.add_module("FC1", nn.Linear(512 * 21 * 21, 32768))
        # self.fc.add_module("BN1", nn.BatchNorm1d(32768, affine=True))
        # self.fc.add_module("Act_layer1", nn.ReLU())
        #
        # self.fc.add_module("FC2", nn.Linear(32768, 2048))
        # self.fc.add_module("BN2", nn.BatchNorm1d(2048, affine=True))
        # self.fc.add_module("Act_layer2", nn.ReLU())

        self.spp_layer = SPPLayer(3, "max_pool")

        self.fc = nn.Sequential()
        self.fc.add_module("FC1", nn.Linear(1024 * 21, 2048))
        self.fc.add_module("BN1", nn.BatchNorm1d(2048))
        self.fc.add_module("Act_layer1", nn.ReLU())

        self.fc.add_module("FC3", nn.Linear(2048, 98 * 2))

    def forward(self, batch_x):
        res = []
        for x in batch_x:
            if torch.cuda.is_available():
                x = x.cuda()
            x = x.unsqueeze(0)
            x = self.conv_layer(x)
            x = self.spp_layer(x)
            res.append(x.squeeze())
        x = torch.stack(res, 0)
        x = self.fc(x)
        return x


class SPPLayer(nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            split_num = 2 ** i
            if h // split_num == 0:
                bottom = split_num - h
                x = nn.ZeroPad2d((0, 0, 0, bottom))(x)
                h = split_num
            if w // split_num == 0:
                right = split_num - w
                x = nn.ZeroPad2d((0, right, 0, 0))(x)
                w = split_num
            kernel_size_h = floor(h / split_num) + (h % split_num)
            stride_h = floor(h / split_num)
            kernel_size_w = floor(w / split_num) + (w % split_num)
            stride_w = floor(w / split_num)
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=(kernel_size_h, kernel_size_w),
                                      stride=(stride_h, stride_w))
                tensor = tensor.view(bs, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=(kernel_size_h, kernel_size_w),
                                      stride=(stride_h, stride_w))
                tensor = tensor.view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x
