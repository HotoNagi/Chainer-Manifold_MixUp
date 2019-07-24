Chainer-Manifold_MixUp
===



Manifold Mixup
---
入力層ではなく隠れ層における表現でmixupを行う。
隠れ層でmixupを行うことで従来よりも良い精度になったという。

論文は[こちら](https://arxiv.org/abs/1806.05236)。
分かりやすい解説は[ここの記事](https://qiita.com/kirikei/items/1fb56f22f4f48c5f91f0)。

コードのおおざっぱな説明
---
chainerでManifold MixUp実装してます。
モデルはとりあえずresnet50だけで試しています。

主にResNet50モデルのforward部分を大幅にいじりました。
論文著者による実装（pytorch）を参考にして書きました。

* model/resnet_fine_manifoldmixup.py
```python
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

```

分類に使用する時はtrain_manifoldmixup.pyで以下のようにManifoldMixupClassiferで包みます。
* train_manifoldmixup.py


```python
model = ManifoldMixupClassifer(ResNet50_Manifold_Mixup(),
                        	mixup_hidden=True, mixup_alpha=0.2,
                                layer_mix=None)

```

結果
---
今回はCIFAR10の学習で比較してみた（α=0.2）。

- [ ] Manifold Mixupなし
![](https://i.imgur.com/EsJ2VxN.png)
validation accuracy : 0.78291

- [ ] Manifold Mixupあり
![](https://i.imgur.com/kTuyJaI.png)
validation accuracy : 0.814062

###### tags: `Mixup` `Classification` `chainer`
