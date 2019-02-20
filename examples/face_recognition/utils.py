# @File  : utils.py
# @Author: X.Yang&Xt.Hao
# @Contact : pistonyang@gmail.com, haoxintongpku@gmail.com
# @Date  : 18-11-1
import sklearn
from mxnet import gluon, autograd as ag
from mxnet.gluon.data.vision import transforms
from gluonfr.metrics.verification import FaceVerification
from mxnet.gluon import nn

def inf_train_gen(loader):
    """
    Using iterations train network.
    :param loader: Dataloader
    :return: batch of data
    """
    while True:
        for batch in loader:
            yield batch


# normalize method commonly used in face recognition.
# operate on an image in the range [0, 255], subtract 127.5 and divided by 128
# which is equal to Normalize a tensorized image in the range [0, 1)
# with mean value 0.5=127.5/225 and std value 128/255
# the class FaceTypeNormalizeTransform and face_type_normalize is almost equal
# the tiny difference is due to numerical calculation
class FaceTypeNormalizeTransform(nn.HybridBlock):
    def __init__(self):
        super(FaceTypeNormalizeTransform, self).__init__()

    def hybrid_forward(self, F, x):
        return (x*255-127.5)*0.0078125


face_type_normalize = transforms.Normalize(0.5, 128/255)

transform_test = transforms.Compose([
    transforms.ToTensor()
])

_transform_train = transforms.Compose([
    transforms.RandomBrightness(0.3),
    transforms.RandomContrast(0.3),
    transforms.RandomSaturation(0.3),
    transforms.RandomFlipLeftRight(),
    transforms.ToTensor()
])


def transform_train(data, label):
    im = _transform_train(data)
    return im, label


class Transform:
    def __init__(self, use_float16=False):
        self._transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

        self._transform_train = transforms.Compose([
            transforms.RandomBrightness(0.3),
            transforms.RandomContrast(0.3),
            transforms.RandomSaturation(0.3),
            transforms.RandomFlipLeftRight(),
            transforms.ToTensor()
        ])
        self.use_float16 = use_float16

    def transform_train(self, data, label):
        im = self._transform_train(data)
        if self.use_float16:
            im = im.astype('float16')
        return im, label

    def transform_test(self, data):
        im = self._transform_test(data)
        if self.use_float16:
            im = im.astype('float16')
        return im


def validate(logger, net, ctx, val_datas, targets, epoch, it, nfolds=10, norm=True):
    metric = FaceVerification(nfolds)
    results = []
    for loader, name in zip(val_datas, targets):
        metric.reset()
        for i, batch in enumerate(loader):
            data0s = gluon.utils.split_and_load(batch[0][0], ctx, even_split=False)
            data1s = gluon.utils.split_and_load(batch[0][1], ctx, even_split=False)
            issame_list = gluon.utils.split_and_load(batch[1], ctx, even_split=False)

            embedding0s = [net(X)[0] for X in data0s]
            embedding1s = [net(X)[0] for X in data1s]
            if norm:
                embedding0s = [sklearn.preprocessing.normalize(e.asnumpy()) for e in embedding0s]
                embedding1s = [sklearn.preprocessing.normalize(e.asnumpy()) for e in embedding1s]

            for embedding0, embedding1, issame in zip(embedding0s, embedding1s, issame_list):
                metric.update(issame, embedding0, embedding1)

        tpr, fpr, accuracy, val, val_std, far, accuracy_std = metric.get()
        logger.info("[{}] Epoch[{}] Batch[{}]: {:.6f}+-{:.6f}".format(name, epoch, it, accuracy, accuracy_std))
        results.append(accuracy)
    return results
