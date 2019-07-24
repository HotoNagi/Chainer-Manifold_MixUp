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


class ResNet50_Manifold_Mixup(chainer.Chain):
    def __init__(self, pretrained_model='../.chainer/dataset/pfnet/chainer/models/ResNet-50-model.npz'):
        super(ResNet50_Manifold_Mixup, self).__init__()
        with self.init_scope():
            self.base = BaseResNet50()
            self.fc6 = L.Linear(2048, 10)
            npz.load_npz(pretrained_model, self.base)

    def forward(self, x, target=None, mixup_hidden=False, lam=1., layer_mix=None):
        if mixup_hidden:
            h , t_a, t_b = self.base(x, target, mixup_hidden, lam, layer_mix)
            #h = F.average_pooling_2d(h, 7, stride=1)
            h = _global_average_pooling_2d(h)
            h = self.fc6(h)

            return h, t_a, t_b
        else:
            h = self.base(x, target, mixup_hidden, lam, layer_mix)
            #h = F.average_pooling_2d(h, 7, stride=1)
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


    def forward(self, x, target, mixup_hidden, lam, layer_mix):
        """
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
            
        return h
        """
        if mixup_hidden:

            if layer_mix == None:
                layer_mix = random.randint(0, 5)

            h = x

            if layer_mix == 0:
                h, t_a, t_b = mixup_data(h, target, lam)            

            h = self.bn1(self.conv1(h))
            h = F.max_pooling_2d(F.relu(h), 3, stride=2)

            if layer_mix == 1:
                h, t_a, t_b = mixup_data(h, target, lam)            

            h = self.res2(h)

            if layer_mix == 2:
                h, t_a, t_b = mixup_data(h, target, lam)

            h = self.res3(h)

            if layer_mix == 3:
                h, t_a, t_b = mixup_data(h, target, lam)

            h = self.res4(h)

            if layer_mix == 4:
                h, t_a, t_b = mixup_data(h, target, lam)

            h = self.res5(h)

            if layer_mix == 5:
                h, t_a, t_b = mixup_data(h, target, lam)


            return h, t_a, t_b

        else:

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

def mixup_data(x, t, lam):

    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    t_a, t_b = t, t[index]
    return mixed_x, t_a, t_b


class ManifoldMixupClassifer(L.Classifier):
    def __init__(self, predictor,
                 lossfun=F.softmax_cross_entropy,
                 accfun=F.accuracy,
                 mixup_hidden=False,  mixup_alpha=0.2,
                 layer_mix=None):
        super().__init__(predictor=predictor, lossfun=lossfun, accfun=accfun)
        self.predictor = predictor
        self.mixup_hidden = mixup_hidden
        self.mixup_alpha = mixup_alpha
        self.layer_mix = layer_mix

    def forward(self, x, t):
        if not chainer.config.train:
            self.y = self.predictor(x=x, target=t, mixup_hidden=False, lam=None, layer_mix=self.layer_mix)
            self.loss = self.lossfun(self.y, t)
        else:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            self.y, t_a, t_b = self.predictor(x=x, target=t, mixup_hidden=self.mixup_hidden, lam=lam, layer_mix=self.layer_mix)
            self.loss = lam * self.lossfun(self.y, t_a) + (1 - lam) * self.lossfun(self.y, t_b)

        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss