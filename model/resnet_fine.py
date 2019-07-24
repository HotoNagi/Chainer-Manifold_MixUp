import chainer
import chainer.functions as F
import chainer.links as L
from chainer.serializers import npz
import chainer.links.model.vision.resnet as R
from chainer import initializers
from chainer.functions.array.reshape import reshape
from chainer import reporter

import random
import numpy as np
import math



class ResNet50_Fine(chainer.Chain):
    def __init__(self, pretrained_model='../.chainer/dataset/pfnet/chainer/models/ResNet-50-model.npz'):
        super(ResNet50_Fine, self).__init__()
        with self.init_scope():
            self.base = BaseResNet50()
            self.fc6 = L.Linear(2048, 10)
            npz.load_npz(pretrained_model, self.base)

    def forward(self, x):
        h = self.base(x)
        h = _global_average_pooling_2d(h)
        h = self.fc6(h)

        return h


class BaseResNet50(chainer.Chain):

    def __init__(self):
        super(BaseResNet50, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 64, 7, 2, 3)
            self.bn1 = L.BatchNormalization(64)
            self.res2 = R.BuildingBlock(3, 64, 64, 256, 1)
            self.res3 = R.BuildingBlock(4, 256, 128, 512, 2)
            self.res4 = R.BuildingBlock(6, 512, 256, 1024, 2)
            self.res5 = R.BuildingBlock(3, 1024, 512, 2048, 2)


    def forward(self, x):
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
            
        return h


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = reshape(h, (n, channel))
    return h