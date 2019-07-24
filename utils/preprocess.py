# -*- coding: utf-8 -*-
import numpy as np
import cupy as cp
import chainer
import chainercv
from chainercv import transforms
from chainercv.datasets import DirectoryParsingLabelDataset
import random
import math


# train用
def _train_transform(data):
    # dataには[(img1,label1),(img2,label2)],...,(imgn,labeln)]みたいな感じで
    # 画像とラベルのlistがタプルにまとまって入っている
    # そのdataを一旦imgとdataに分割
    img, lable = data
    # ランダム回転することでデータ数を水増しする
    img = chainercv.transforms.random_rotate(img)
    # ランダムフリップ（反転）することでデータ数を水増しする
    img = chainercv.transforms.random_flip(img, x_random=True, y_random=True)
    return img, lable


# validation用
def _validation_transform(data):
    # dataには[(img1,label1),(img2,label2)],...,(imgn,labeln)]みたいな感じで
    # 画像とラベルのlistがタプルにまとまって入っている
    # そのdataを一旦imgとdataに分割
    img, lable = data
    # 画像を正方形に変形する（高さ、幅どちらか小さいほうに合わせるっぽい）
    img = chainercv.transforms.scale(img, 256)
    # 中心を指定サイズで切り抜く
    img = chainercv.transforms.center_crop(img, (224, 224))
    return img, lable