# MIT License
#
# Copyright (c) 2018 Haoxintong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
""""""


import argparse, time, logging, os, math, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import mxnet as mx
import numpy as np
from tqdm import tqdm
from mxnet import gluon, autograd as ag
from mxnet.gluon.data import DataLoader
from datetime import datetime
from gluonfr.utils.lr_scheduler import LRScheduler
from gluonfr.loss import *
from gluonfr.model_zoo import *
from gluonfr.data import get_recognition_dataset
from examples.face_recognition.utils import transform_test, transform_train, validate


# CLI
parser = argparse.ArgumentParser(description='Train a face recognition model.')
parser.add_argument('--data-dir', type=str, default='/media/deep/t6/datasets/insightface',
                    help='training and validation data to use.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training. default is float32')
parser.add_argument('--gpus', type=str, default='4,5,6,7',
                    help='id of gpus to use.')
parser.add_argument('--log-dir', type=str, default='./log',
                    help='folder to save log')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=24, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--num-epochs', type=int, default=40,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.1.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0005,
                    help='weight decay rate. default is 0.0005.')
parser.add_argument('--lr-mode', type=str, default='step',
                    help='learning rate scheduler mode. options are step, poly and cosine.')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--lr-decay-period', type=int, default=0,
                    help='interval for periodic learning rate decays. default is 0 to disable.')
parser.add_argument('--lr-decay-epoch', type=str, default='9,15,21',
                    help='epochs at which learning rate decays. default is (9,15,21).')
parser.add_argument('--warmup-lr', type=float, default=0.0,
                    help='starting warmup learning rate. default is 0.0.')
parser.add_argument('--warmup-epochs', type=int, default=0,
                    help='number of warmup epochs.')
parser.add_argument('--last-gamma', action='store_true',
                    help='whether to init gamma of the last BN layer in each bottleneck to 0.')
parser.add_argument('--mode', type=str, default='hybrid',
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')
parser.add_argument('--model', type=str,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--input-size', type=int, default=112,
                    help='size of the input image size. default is 112')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--use_se', action='store_true',
                    help='use SE layers or not in resnext. default is false.')
parser.add_argument('--mixup', action='store_true',
                    help='whether train the model with mix-up. default is false.')
parser.add_argument('--mixup-alpha', type=float, default=0.2,
                    help='beta distribution parameter for mixup sampling, default is 0.2.')
parser.add_argument('--mixup-off-epoch', type=int, default=0,
                    help='how many last epochs to train without mixup, default is 0.')
parser.add_argument('--label-smoothing', action='store_true',
                    help='use label smoothing or not in training. default is false.')
parser.add_argument('--no-wd', action='store_true',
                    help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
parser.add_argument('--save-frequency', type=int, default=1000,
                    help='frequency of model saving. default is 1000.')
parser.add_argument('--save-dir', type=str, default='./params',
                    help='directory of saved models')
parser.add_argument('--resume-epoch', type=int, default=0,
                    help='epoch to resume training from.')
parser.add_argument('--resume-params', type=str, default='',
                    help='path of parameters to load from.')
parser.add_argument('--resume-states', type=str, default='',
                    help='path of trainer state to load from.')
parser.add_argument('--log-interval', type=int, default=50,
                    help='Number of batches to wait before logging.')
parser.add_argument('--logging-file', type=str, default='train.log',
                    help='name of training log file.')
parser.add_argument('--margin1', type=float, default=1.0,
                    help='m1 in combined margin loss.')
parser.add_argument('--margin2', type=float, default=0.3,
                    help='m2 in combined margin loss.')
parser.add_argument('--margin3', type=float, default=0.2,
                    help='m3 in combined margin loss.')
parser.add_argument('--scale', type=float, default=64,
                    help='s in margin loss.')
parser.add_argument('--embedding-size', type=int, default=512,
                    help='embedding vector size.')
parser.add_argument('--validation-targets', type=str, default='lfw,cfp_fp,agedb_30')
opt = parser.parse_args()

# directory to save logs and models
if not os.path.exists(os.path.join(opt.log_dir)):
    os.makedirs(os.path.join(opt.log_dir))
if not os.path.exists(os.path.join(opt.save_dir)):
    os.makedirs(os.path.join(opt.save_dir))

# set up for logging
filehandler = logging.FileHandler(os.path.join(opt.log_dir, "resnet50v1d-combinedloss%s.log" % datetime.strftime(datetime.now(), '%m%d_%H')))
streamhandler = logging.StreamHandler()
logger = logging.getLogger('TRAIN')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

logger.info(opt)

# context setting
if opt.gpus:
    gpu_list = opt.gpus.split(',')
    num_gpu = len(gpu_list)
else:
    num_gpu = 0
if num_gpu:
    ctx = [mx.gpu(int(i)) for i in gpu_list]
    logger.info('use %d gpus'%num_gpu)
    logger.info(ctx)
else:
    ctx = [mx.cpu()]
    logger.info('use cpu')
    logger.info(ctx)

batch_size = (opt.batch_size * num_gpu) if num_gpu > 0 else opt.batch_size
lr_decay_epoch = [int(i)+opt.warmup_epochs for i in opt.lr_decay_epoch.split(',')]

logger.info('loading training and validation data')
image_src_root = opt.data_dir
train_set = get_recognition_dataset("faces_emore", root=image_src_root, transform=transform_train)
train_data = DataLoader(train_set, batch_size, shuffle=True, num_workers=opt.num_workers)

# targets = ['lfw', 'cfp_fp', 'agedb_30']
targets = [i for i in opt.validation_targets.split(',')]
val_sets = [get_recognition_dataset(name, root=os.path.join(image_src_root, 'faces_emore'), transform=transform_test) for name in targets]
val_datas = [DataLoader(dataset, batch_size, num_workers=opt.num_workers) for dataset in val_sets]

logger.info('-------------------- net structure --------------------')
net = face_resnet50_v1d(ctx=ctx, classes=train_set.num_classes,
                        embedding_size=opt.embedding_size, activation_type='prelu',
                        weight_norm=True, feature_norm=True,
                        norm_kwargs={'epsilon':2e-5})
net.initialize(init=mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2), ctx=ctx)
# net.initialize(init=mx.init.MSRAPrelu())
# net.load_parameters("models/mobilefacenet-ring-it-185000.params", ctx=ctx)
if opt.mode == 'hybrid':
    net.hybridize(static_alloc=True, static_shape=True)
logger.info(net)
loss = CombinedLoss(train_set.num_classes, m1=opt.margin1, m2=opt.margin2, m3=opt.margin3, s=opt.scale, easy_margin=False)
if opt.mode == 'hybrid':
    loss.hybridize(static_alloc=True, static_shape=True)
logger.info(loss)
logger.info('-------------------------------------------------------')

train_params = net.collect_params()
train_params.update(loss.params)
batches_per_epoch = len(train_set) // batch_size
lr_scheduler = LRScheduler(mode=opt.lr_mode, baselr=opt.lr, niters=batches_per_epoch,
                           nepochs=opt.num_epochs, step=lr_decay_epoch, step_factor=0.1,
                           warmup_epochs=opt.warmup_epochs, warmup_lr=opt.warmup_lr)
_scale = 1.0 / num_gpu
optimizer_params = {'momentum': opt.momentum,
                    'wd': opt.wd,
                    'lr_scheduler': lr_scheduler
                   }
# multi precision training
if opt.dtype != 'float32':
    optimizer_params['multi_precision'] = True
trainer = gluon.Trainer(train_params, 'sgd', optimizer_params)
it = 0

loss_mtc, acc_mtc = mx.metric.Loss(), mx.metric.Accuracy()
tic = time.time()
highest_acc = [0.0, 0.0]
for epoch in range(opt.resume_epoch, opt.num_epochs):

    for i, batch in enumerate(tqdm(train_data)):
        it = epoch * batches_per_epoch + i
        lr_scheduler.update(i, epoch)

        datas = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

        with ag.record():
            ots = [net(X) for X in datas]
            outputs = [ot[1] for ot in ots]
            losses = [loss(yhat, y) for yhat, y in zip(outputs, labels)]

        for l in losses:
            ag.backward(l)

        trainer.step(batch_size)
        acc_mtc.update(labels, outputs)
        loss_mtc.update(0, losses)

        if (it % opt.save_frequency) == 0:
            _, train_loss = loss_mtc.get()
            train_metric_name, train_acc = acc_mtc.get()
            toc = time.time()

            logger.info('\nEpoch[%d] Batch[%d]\tSpeed: %f samples/sec\t%s=%f\tlr=%f train loss: %.6f' %
                        (epoch, it, batch_size*opt.save_frequency/(toc-tic), train_metric_name,train_acc, trainer.learning_rate, train_loss))

            acc_list = validate(logger, net, ctx, val_datas, targets, epoch, it)

            # save flags
            do_save = False
            is_highest = False
            # check if save or not
            if len(acc_list) > 0:
                score = sum(acc_list)
                if acc_list[-1] >= highest_acc[-1]:
                    if acc_list[-1] > highest_acc[-1]:
                        is_highest = True
                    else:
                        if score >= highest_acc[0]:
                            is_highest = True
                            highest_acc[0] = score
                    highest_acc[-1] = acc_list[-1]
                    # if lfw_score>=0.99:
                    #  do_save = True
            if is_highest:
                do_save = True
            loss_mtc.reset()
            acc_mtc.reset()
            tic = time.time()
            if do_save:
                net.save_parameters(os.path.join(opt.save_dir, "resnet50v1d-combined-%d.params" % it))
        it += 1
    epoch += 1
