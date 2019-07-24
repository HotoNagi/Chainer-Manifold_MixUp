# -*- coding: utf-8 -*-

import chainer
from chainer.training import extensions
from chainer.datasets import get_cifar10
from chainer import training

from utils.preprocess import _train_transform
# from utils.preprocess import _validation_transform

import matplotlib.pyplot as plt
plt.switch_backend('agg')  # これがないとmatplotlibでバグる

from model.resnet_fine_manifoldmixup import ResNet50_Manifold_Mixup
from model.resnet_fine_manifoldmixup import ManifoldMixupClassifer


def main():

    train, validation = get_cifar10()
    train = chainer.datasets.TransformDataset(train, _train_transform)
    # validation = chainer.datasets.TransformDataset(validation, _validation_transform)
    gpu_device = 0  # CPUを使いたい場合は-1を指定し
    epoch = 300  # エポック数を指定
    batch_size = 256  # バッチサイズを指定
    frequency = -1  # Take a snapshot for each specified epoch

    model = ManifoldMixupClassifer(ResNet50_Manifold_Mixup(),
                                   mixup_hidden=True, mixup_alpha=0.2,
                                   layer_mix=None)

    if gpu_device >= 0:
        chainer.backends.cuda.get_device_from_id(gpu_device).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.MultithreadIterator(train,
                                                       batch_size,
                                                       n_threads=8)
    validation_iter = chainer.iterators.MultithreadIterator(validation,
                                                            batch_size,
                                                            repeat=False,
                                                            shuffle=False,
                                                            n_threads=8)

    updater = training.StandardUpdater(train_iter,
                                       optimizer,
                                       device=gpu_device)

    trainer = training.Trainer(updater, (epoch, 'epoch'))
    trainer.extend(extensions.Evaluator(validation_iter,
                                        model,
                                        device=gpu_device))
    trainer.extend(extensions.dump_graph('main/loss'))
    frequency = epoch if frequency == -1 else max(1, frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                         'epoch', file_name='lossmixup.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                                         'epoch', file_name='accuracymixup.png'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',
                                           'main/accuracy', 'validation/main/accuracy',
                                           'elapsed_time']))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()


if __name__ == '__main__':
    main()
