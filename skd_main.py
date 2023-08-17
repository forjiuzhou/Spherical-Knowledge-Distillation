import argparse
import os
import shutil
import time
import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import nvidia.dali.tfrecord as tfrec
import torch.nn.functional as F
import numpy as np

from torch.utils.tensorboard import SummaryWriter

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    import nvidia.dali as dali
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='*',
                    help='path(s) to dataset (if one path is provided, it is assumed\n' +
                    'to have subdirectories named "train" and "val"; alternatively,\n' +
                    'train and val paths can be specified directly by providing both paths as arguments)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--fp16', action='store_true',
                    help='Run model fp16 mode.')
parser.add_argument('--dali_cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                    '--static-loss-scale.')
parser.add_argument('--prof', dest='prof', action='store_true',
                    help='Only run 10 iterations for profiling.')
parser.add_argument('-t', '--test', action='store_true',
                    help='Launch test mode with preset arguments')

parser.add_argument("--local_rank", default=0, type=int)

# parser.add_argument("--model_dir", type=str)
parser.add_argument('--T', type=float)
parser.add_argument('--alpha', type=float)
parser.add_argument('--multiplier', type=float)
parser.add_argument('--distillation', action='store_true')
parser.add_argument('--SKD', action='store_true')
parser.add_argument('--log_str', type=str)

cudnn.benchmark = True


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        # self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=True)
        index_path = []
        for path in os.listdir("/home/yz979/code/imagenet/ImageNet-to-TFrecord/idx_files"):
            index_path.append(os.path.join("/home/yz979/code/imagenet/ImageNet-to-TFrecord/idx_files", path))
        index_path = sorted(index_path)
        self.input = ops.TFRecordReader(path=data_dir, index_path=index_path, shard_id=args.local_rank,
                                        num_shards=args.world_size, random_shuffle=True,
                                        features={
                                                    'image/height': tfrec.FixedLenFeature([1], tfrec.int64,  -1),
                                                    'image/width': tfrec.FixedLenFeature([1], tfrec.int64,  -1),
                                                    'image/colorspace': tfrec.FixedLenFeature([ ], tfrec.string, ''),
                                                    'image/channels': tfrec.FixedLenFeature([], tfrec.int64,  -1),
                                                    'image/class/label': tfrec.FixedLenFeature([1], tfrec.int64,  -1),
                                                    'image/class/synset': tfrec.FixedLenFeature([ ], tfrec.string, ''),
                                                    # 'image/class/text': tfrec.FixedLenFeature([ ], tfrec.string, ''),
                                                    # 'image/object/bbox/xmin': tfrec.VarLenFeature(tfrec.float32, 0.0),
                                                    # 'image/object/bbox/xmax': tfrec.VarLenFeature(tfrec.float32, 0.0),
                                                    # 'image/object/bbox/ymin': tfrec.VarLenFeature(tfrec.float32, 0.0),
                                                    # 'image/object/bbox/ymax': tfrec.VarLenFeature(tfrec.float32, 0.0),
                                                    # 'image/object/bbox/label': tfrec.FixedLenFeature([1], tfrec.int64,-1),
                                                    'image/format': tfrec.FixedLenFeature((), tfrec.string, ""),
                                                    'image/filename': tfrec.FixedLenFeature((), tfrec.string, ""),
                                                    'image/encoded': tfrec.FixedLenFeature((), tfrec.string, "")
                                                })

        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        inputs = self.input()
        self.jpegs = inputs["image/encoded"]
        self.labels = inputs["image/class/label"]
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        # self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=False)
        index_path = []
        for path in os.listdir("/home/yz979/code/imagenet/ImageNet-to-TFrecord/val_idx_files"):
            index_path.append(os.path.join("/home/yz979/code/imagenet/ImageNet-to-TFrecord/val_idx_files", path))
        index_path = sorted(index_path)
        self.input = ops.TFRecordReader(path=data_dir, index_path=index_path, shard_id=args.local_rank,
                                        num_shards=args.world_size, random_shuffle=True,
                                        features={
                                            'image/height': tfrec.FixedLenFeature([1], tfrec.int64, -1),
                                            'image/width': tfrec.FixedLenFeature([1], tfrec.int64, -1),
                                            'image/colorspace': tfrec.FixedLenFeature([], tfrec.string, ''),
                                            'image/channels': tfrec.FixedLenFeature([], tfrec.int64, -1),
                                            'image/class/label': tfrec.FixedLenFeature([1], tfrec.int64, -1),
                                            'image/class/synset': tfrec.FixedLenFeature([], tfrec.string, ''),
                                            'image/format': tfrec.FixedLenFeature((), tfrec.string, ""),
                                            'image/filename': tfrec.FixedLenFeature((), tfrec.string, ""),
                                            'image/encoded': tfrec.FixedLenFeature((), tfrec.string, "")
                                        })
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        # self.jpegs, self.labels = self.input(name="Reader")

        inputs = self.input()
        self.jpegs = inputs["image/encoded"]
        self.labels = inputs["image/class/label"]

        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


best_prec1 = 0
best_prec5 = 0
args = parser.parse_args()
if args.local_rank == 0:
    writer = SummaryWriter(log_dir=args.log_str)

# test mode, use default args for sanity test
if args.test:
    args.fp16 = False
    args.epochs = 1
    args.start_epoch = 0
    args.arch = 'resnet50'
    args.batch_size = 64
    args.data = []
    args.prof = True
    args.data.append('/data/imagenet/train-jpeg/')
    args.data.append('/data/imagenet/val-jpeg/')

if not len(args.data):
    raise Exception("error: too few arguments")

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) >= 1

# make apex optional
if args.fp16 or args.distributed:
    try:
        from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import *
        from apex import *
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


# dali.backend.SetHostBufferShrinkThreshold(1)


def main():
    global best_prec1, args, best_prec5
    print("         A")
    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    teacher = None
    if args.distillation:
        teacher = models.resnet50(pretrained=True).cuda()
        teacher.eval()

    from torchvision.models.resnet import resnet18
    checkpoint = 'https://github.com/forjiuzhou/Spherical-Knowledge-Distillation/releases/download/v1/resnet18_skd.pth'
    model = resnet18()
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False, map_location="cpu", check_hash=True))

    model = model.cuda()

    if args.distributed:
        # shared param/delay all reduce turns off bucketing in DDP, for lower latency runs this can improve perf
        # for the older version of APEX please use shared_param, for newer one it is delay_allreduce
        model = DDP(model, delay_allreduce=True)
        if args.distillation:
            teacher = DDP(teacher, delay_allreduce=True)
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        if args.distillation:
            teacher = amp.initialize(teacher, opt_level="O1")
    print("         C")
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    # Data loading code
    if len(args.data) == 1:
        traindir = []
        valdir = []
        for path in os.listdir(os.path.join(args.data[0], "train")):
            traindir.append(os.path.join(args.data[0],"train", path))
        traindir = sorted(traindir)
        for path in os.listdir(os.path.join(args.data[0], "validation")):
            valdir.append(os.path.join(args.data[0], "validation",path))
        valdir = sorted(valdir)
        print(len(valdir), len(traindir))
    else:
        traindir = args.data[0]
        valdir= args.data[1]

    if(args.arch == "inception_v3"):
        crop_size = 299
        val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224
        val_size = 256

    pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=traindir, crop=crop_size, dali_cpu=args.dali_cpu)
    pipe.build()

    train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size()['__TFRecordReader_1'] / args.world_size))
    pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=valdir,
                         crop=crop_size, size=val_size)
    pipe.build()

    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size()['__TFRecordReader_5'] / args.world_size))

    if args.evaluate:
        validate(train_loader, model, criterion, teacher)
        return
    total_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        if epoch < 30:
            args.alpha = 0.9
        elif epoch < 60:
            args.alpha = 0.9
        elif epoch < 80:
            args.alpha = 0.5
        elif epoch < 100:
            args.alpha = 0.1
        if epoch == 30 or epoch == 60 or epoch == 90:
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, False, filename=os.path.join(args.log_str, 'epoch_{}'.format(epoch) + '_checkpoint.pth.tar'))
        # train for one epoch
        loss_kd = None
        if args.distillation:
            avg_train_time, losses, top1, top5, loss_kd, \
                = train_kd(train_loader, model, (teacher,None), criterion, optimizer, epoch)
        else:
            avg_train_time, losses, top1, top5 = train(train_loader, model, criterion, optimizer, epoch)
        total_time.update(avg_train_time)
        if args.local_rank == 0:
            writer.add_scalar('Loss/train', losses, epoch)
            writer.add_scalar('Accuracy/train_prec1', top1, epoch)
            writer.add_scalar('Accuracy/train_prec5', top5, epoch)

            if loss_kd:
                writer.add_scalar('Loss/train/kd_loss', loss_kd, epoch)
        if args.prof:
            break
        # evaluate on validation set

        with torch.no_grad():
            prec1, prec5, losses, loss_kd = validate(val_loader, model, criterion, teacher)

            if args.local_rank == 0:
                writer.add_scalar('Loss/test', losses, epoch)
                writer.add_scalar('Accuracy/test_prec1', prec1, epoch)
                writer.add_scalar('Accuracy/test_prec5', prec5, epoch)
                if loss_kd:
                    writer.add_scalar('Loss/test/loss_kd', loss_kd, epoch)
        torch.cuda.empty_cache()
        # val_pipe.release_outputs()
        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            best_prec5 = max(prec5, best_prec5)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                'best_prec5': best_prec5,
            }, is_best)
            if epoch == args.epochs - 1:
                print('##Top-1 {0}\n'
                      '##Top-5 {1}\n'
                      '##Perf  {2}'.format(prec1, prec5, args.total_batch_size / total_time.avg))

        # reset DALI iterators
        del prec5, prec1, losses
        train_loader.reset()
        val_loader.reset()

def train_kd(train_loader, model, teacher, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses_kd = AverageMeter()
    losses_softmax = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):

        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long() - 1
        train_loader_len = int(math.ceil(train_loader._size / args.batch_size))

        adjust_learning_rate(optimizer, epoch, i, train_loader_len)

        if args.prof:
            if i > 10:
                break
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        output = model(input_var)
        with torch.no_grad():
            output_t = teacher[0](input_var)

        if args.SKD:
            # output = F.layer_norm(
            #     output, torch.Size((1000,)), None, None, 1e-7)*args.multiplier
            # output_t = F.layer_norm(
            #     output_t, torch.Size((1000,)), None, None, 1e-7)*args.multiplier
            # p = F.softmax(output_t / args.T, dim=-1)
            # loss_kd = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(output / args.T, dim=1),p
                                                    #   ) * args.T * args.T * args.alpha
            tea_std = torch.std(output_t, dim=-1,keepdim=True)
            stu_std= torch.std(output, dim=-1, keepdim=True)
            p_s = F.log_softmax(output/stu_std*tea_std/args.T, dim=1)
            p_t = F.softmax(output_t/args.T, dim=1)
            loss_kd = torch.sum(torch.sum(F.kl_div(p_s, p_t, reduction='none'), dim=-1) * (args.T* args.T * torch.ones(output.shape[0],1).cuda())) /output.shape[0]/ output.shape[0] * args.alpha
            output = output/stu_std*tea_std

        loss_softmax = criterion(output, target_var) * (1-args.alpha)

        loss = loss_kd+loss_softmax
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            reduced_loss_kd = reduce_tensor(loss_kd.data)
            reduced_softmax = reduce_tensor(loss_softmax.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)

        else:
            reduced_loss_kd = loss_kd.data
            reduced_softmax = loss_softmax.data
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))
        losses_kd.update(to_python_float(reduced_loss_kd), input.size(0))
        losses_softmax.update(to_python_float(reduced_softmax), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        else:
            loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0 and i > 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_softmax {loss_softmax.val:.4f} ({loss_softmax.avg:.4f})\t'
                  'Loss_kd {loss_kd.val:.4f} ({loss_kd.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, train_loader_len,
                args.total_batch_size / batch_time.val,
                args.total_batch_size / batch_time.avg,
                batch_time=batch_time,
                data_time=data_time, loss=losses, loss_kd=losses_kd, loss_softmax=losses_softmax, top1=top1, top5=top5))
        del loss, output
    return batch_time.avg, losses.avg, top1.avg, top5.avg, losses_kd.avg


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()-1
        train_loader_len = int(math.ceil(train_loader._size / args.batch_size))

        adjust_learning_rate(optimizer, epoch, i, train_loader_len)

        if args.prof:
            if i > 10:
                break
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        output = model(input_var)

        if args.SKD:
            output = F.layer_norm(
                output, torch.Size((1000,)), None, None, 1e-9) * args.multiplier

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0 and i > 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, train_loader_len,
                   args.total_batch_size / batch_time.val,
                   args.total_batch_size / batch_time.avg,
                   batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
        del loss, output
    return batch_time.avg, losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, teacher=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_kd = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()-1
        val_loader_len = int(val_loader._size / args.batch_size)

        target = target.cuda(non_blocking=True)
        input = Variable(input)
        target = Variable(target)
        # compute output
        with torch.no_grad():
            output = model(input)

            if args.SKD:
                output = F.layer_norm(
                    output, torch.Size((1000,)), None, None, 1e-9) * args.multiplier
            loss = criterion(output, target)

            if args.distillation:
                output_t = teacher(input)
                if args.SKD:
                    output_t = F.layer_norm(
                        output_t, torch.Size((1000,)), None, None, 1e-9) * args.multiplier
                loss_kd = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(output / args.T, dim=1),
                                                              F.softmax(output_t / args.T,
                                                                        dim=-1)) * args.T * args.T


        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            if args.distillation:
                reduced_loss_kd = reduce_tensor(loss_kd.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        if args.distillation:
            losses_kd.update(to_python_float(reduced_loss_kd), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_kd {losses_kd.val:.4f} ({losses_kd.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, val_loader_len,
                   args.total_batch_size / batch_time.val,
                   args.total_batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses, losses_kd=losses_kd,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    if args.distillation:
        loss_kd_avg = losses_kd.avg
    else:
        loss_kd_avg = None
    print("loss_kd:  {}".format(loss_kd_avg))
    print("loss_softmax:   {}".format(losses.avg))
    return top1.avg, top5.avg, losses.avg,loss_kd_avg


def save_checkpoint(state, is_best, filename=os.path.join(args.log_str, 'checkpoint.pth.tar')):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.log_str, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30
    # factor = epoch // 100
    if epoch >= 80:
        factor = factor + 1
    # if epoch >= 90:
    #     factor = factor + 1
    lr = args.lr * (0.1 ** factor)

    """Warmup"""
    # if epoch < 5:
    #     lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    if(args.local_rank == 0 and step % args.print_freq == 0 and step > 1):
        print("Epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    main()