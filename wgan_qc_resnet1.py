from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd
import os
import shutil
from glob import *

from cvxopt import matrix, spmatrix, sparse, solvers
import numpy as np
from copy import deepcopy

import models.resnet1 as resnet


####################################################################################
# WGAN-QC (Wasserstein GANs with Quadratic Transport Cost)
# Code is adopted from WGAN and modified by Huidong Liu (Hui-Dong Liu)
# Email: huidliu@cs.stonybrook.edu; h.d.liew@gmail.com
####################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='mnist | cifar10 | lsun | imagenet | folder | lfw | others')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--ngf_max', type=int, default=512)
parser.add_argument('--ndf_max', type=int, default=512)
parser.add_argument('--K', type=float, default=-1.0, help='the coef in transport cost, <=0 meaning K = 1/dim') 
parser.add_argument('--res_ratio', type=float, default=0.1, help='resnet block ratio') 
parser.add_argument('--epochs', type=int, default=2000, help='number of epochs for training')
parser.add_argument('--epoch', type=int, default=1, help='starting epoch (to continue training)')
parser.add_argument('--Giters', type=int, default=500000, help='number of Generator iterations, default=50k')
parser.add_argument('--Giter', type=int, default=0, help='starting Generator iteration (to continue training)')
parser.add_argument('--lrD', type=float, default=1e-4, help='learning rate for D, default=1e-4')
parser.add_argument('--lrG', type=float, default=1e-4, help='learning rate for G, default=1e-4')
parser.add_argument('--alpha', type=float, default=0.0, help='alpha, default=0.1')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--lr_anneal', type=float, default=1.0, help='learning rate decay rate, default=1.0')
parser.add_argument('--milestones', default='50000,150000', help='milestones of lr anneal.')
parser.add_argument('--gamma', type=float, default=0.1, help='gamma for optimal transport regularization')
parser.add_argument('--EMA', type=float, default=0.999, help='Exponential Moving Average, default=0, range: [0,1)')
parser.add_argument('--EMA_startIter', type=int, default=80000, help='start EMA from which G iter')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disable CUDA training')
parser.add_argument('--retrain', action='store_true', default=False, help='re-train or not')
parser.add_argument('--pin_mem', action='store_true', help='use pin memory or not')
parser.add_argument('--gpu_ids', default='0', help='GPU ids visible to CUDA')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--Diters', type=int, default=1, help='number of D iters')
parser.add_argument('--DOptIters', type=int, default=1, help='number of iters of regression of D, default=1')
parser.add_argument('--output_dir', default=None, help='folder to store output samples and checkpoints')
parser.add_argument('--RMSprop', action='store_true', help='Whether to use RMSprop (default is Adam)')
parser.add_argument('--Adagrad', action='store_true', help='Whether to use Adagrad (default is Adam)')
parser.add_argument('--IS_freq', type=int, default=10000, help='IS evaluation frequency if IS is activated') 
parser.add_argument('--FID_freq', type=int, default=10000, help='FID evaluation frequency if FID is activated')
parser.add_argument('--verbose_freq', type=int, default=10, help='verbose frequency')
parser.add_argument('--genImg_num', type=int, default=64, help='number of generated images') 
parser.add_argument('--genImg_freq', type=int, default=500, help='generate image frequency') 
parser.add_argument('--save_ckpt_freq', type=int, default=10000, help='save checkpoint frequency')
parser.add_argument('--save_ckpt_epoch_freq', type=int, default=10, help='save checkpoint in how many epochs')

args = parser.parse_args()
print(args)

nz = int(args.nz)
ngf = int(args.ngf)
ndf = int(args.ndf)
ngf_max = int(args.ngf_max)
ndf_max = int(args.ndf_max)
K = args.K
res_ratio = args.res_ratio
batchSize = int(args.batchSize)
Diters = int(args.Diters)
DOptIters = int(args.DOptIters)
epochs = int(args.epochs)
epoch = int(args.epoch)
Giters = int(args.Giters)
Giter = int(args.Giter)
alpha = args.alpha
gamma = args.gamma
EMA = args.EMA
EMA_startIter = int(args.EMA_startIter)
gpu_ids = args.gpu_ids
lr_anneal = args.lr_anneal
milestones = list(map(int, args.milestones.split(',')))
IS_freq = int(args.IS_freq)
FID_freq = int(args.FID_freq)
verbose_freq = int(args.verbose_freq)
genImg_freq = int(args.genImg_freq)
genImg_num = int(args.genImg_num)
genImg_num_sroot = int(genImg_num ** 0.5)
save_ckpt_freq = int(args.save_ckpt_freq)
save_ckpt_epoch_freq = int(args.save_ckpt_epoch_freq)

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

output_dir = args.output_dir

if output_dir is None:
    print("WARNING: No output_dir provided. Results will be saved to ./outputs")
    output_dir = './outputs'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    if args.retrain:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

img_path = '{0}/images'.format(output_dir)
checkpoint_path = '{0}/checkpoints'.format(output_dir)
if not os.path.exists(img_path):
    os.makedirs(img_path)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

log_fileName = '{0}/log.txt'.format(output_dir)
with open(log_fileName, 'a') as f:
    f.write('\n{}\n'.format(str(args)))

args.manualSeed = random.randint(1, 10000)  
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

device = torch.device("cuda:{0}".format(0) if torch.cuda.is_available() else "cpu")

args.cuda = not args.no_cuda and torch.cuda.is_available()
if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with cuda")

Optimizer = 'Adam'
if args.RMSprop:
    Optimizer = 'RMSprop'
if args.Adagrad:
    Optimizer = 'Adagrad'

    
class CenterCropLongEdge(object):
  """Crops the given PIL Image on the long edge.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    return transforms.functional.center_crop(img, min(img.size))

  def __repr__(self):
    return self.__class__.__name__


if args.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   CenterCropLongEdge(),
                                   transforms.Resize(args.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc = 3
elif args.dataset == 'lsun':
    dataset = dset.LSUN(root=args.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(args.imageSize),
                            transforms.CenterCrop(args.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc = 3
elif args.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=args.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(args.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
    nc = 3
elif args.dataset == 'mnist':
    dataset = dset.MNIST(root=args.dataroot,
                         train=True, # download=True,
                         transform=transforms.Compose([
                            transforms.Resize(args.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ])
    )
    nc = 1
else:
   dataset = dset.ImageFolder(root=args.dataroot,
                         transform=transforms.Compose([
                            transforms.Resize(args.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ])
    )
   nc = 3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=int(args.workers),
                                         drop_last=True, pin_memory=args.pin_mem)


# custom weights initialization called on netG and netD
def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def weights_init_G(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


netG = resnet.ResNet_G(nz, args.imageSize, nfilter=ngf, nfilter_max=ngf_max, res_ratio=res_ratio)
netD = resnet.ResNet_D(nz, args.imageSize, nfilter=ndf, nfilter_max=ndf_max, res_ratio=res_ratio)

if torch.cuda.device_count() > 1:
  print("use", torch.cuda.device_count(), "GPUs!")
  netG = nn.DataParallel(netG)
  netD = nn.DataParallel(netD)

netG.apply(weights_init_G)
if args.netG != '':  # load checkpoint if needed
    netG.load_state_dict(torch.load(args.netG))
print(netG)

netD.apply(weights_init_D)
if args.netD != '':
    netD.load_state_dict(torch.load(args.netD))
print(netD)

with open(log_fileName, 'a') as f:
    print(netD, file=f)
    print(netG, file=f)

data_dim = nc * args.imageSize * args.imageSize
if K <= 0:
    K = 1.0 / data_dim
Kr = np.sqrt(K)
LAMBDA = 2 * Kr * gamma * 2
real = torch.FloatTensor(batchSize, nc, args.imageSize, args.imageSize).to(device)
noise = torch.FloatTensor(batchSize, nz, 1, 1).to(device)
one = torch.FloatTensor([1])
mone = one * -1
ones = torch.ones(batchSize)
one, mone, ones = one.to(device), mone.to(device), ones.to(device)
netD = netD.to(device)
netG = netG.to(device)

criterion = torch.nn.MSELoss()


def build_optimizerD(Optimizer='Adam'):
    # build optimizer
    if Optimizer == 'RMSprop':
        optimizerD = optim.RMSprop(netD.parameters(), lr = args.lrD)
    elif Optimizer == 'Adam':
        optimizerD = optim.Adam(netD.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
    elif Optimizer == 'Adagrad':
        optimizerD = optim.Adagrad(netD.parameters(), lr=args.lrD)
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr = args.lrD)

    return optimizerD


def build_optimizerG(Optimizer='Adam'):
    # build optimizer
    if Optimizer == 'RMSprop':
        optimizerG = optim.RMSprop(netG.parameters(), lr=args.lrG)
    elif Optimizer == 'Adam':
        optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
    elif Optimizer == 'Adagrad':
        optimizerG = optim.Adagrad(netG.parameters(), lr=args.lrG)
    else:
        optimizerG = optim.RMSprop(netG.parameters(), lr=args.lrG)

    return optimizerG


def build_lr_scheduler(optimizer, milestones, lr_anneal, last_epoch=-1):
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_anneal, last_epoch=-1)
    return scheduler


def load_last_checkpoint(netD, netG, checkpoint_path):
    checkpoints = glob('{}/*.pth'.format(checkpoint_path))
    checkpoint_ids = [(int(f.split('_')[2]), int(f.split('_')[4]), f) for f in [p.split('/')[-1].split('.')[0] for p in checkpoints]]
    if not checkpoint_ids:
        epoch = 1
        Giter = 1
        print('No netD or netG loaded!')
    else:
        epoch, Giter, _ = max(checkpoint_ids, key=lambda item: item[1])
        netD.load_state_dict(torch.load('{}/netD_epoch_{}_Giter_{}.pth'.format(checkpoint_path, epoch, Giter)))
        print('netD_epoch_{}_Giter_{}.pth loaded!'.format(epoch, Giter))
        netG.load_state_dict(torch.load('{}/netG_epoch_{}_Giter_{}.pth'.format(checkpoint_path, epoch, Giter)))
        print('netG_epoch_{}_Giter_{}.pth loaded!'.format(epoch, Giter))
        Giter += 1

    return epoch, Giter


###############################################################################
###################### Prepare linear programming solver ######################
solvers.options['show_progress'] = False
solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}

A = spmatrix(1.0, range(batchSize), [0]*batchSize, (batchSize,batchSize))
for i in range(1,batchSize):
    Ai = spmatrix(1.0, range(batchSize), [i]*batchSize, (batchSize,batchSize))
    A = sparse([A,Ai])

D = spmatrix(-1.0, range(batchSize), range(batchSize), (batchSize,batchSize))
DM = D
for i in range(1,batchSize):
    DM = sparse([DM, D])

A = sparse([[A],[DM]])

cr = matrix([-1.0/batchSize]*batchSize)
cf = matrix([1.0/batchSize]*batchSize)
c = matrix([cr,cf])

pStart = {}
pStart['x'] = matrix([matrix([1.0]*batchSize),matrix([-1.0]*batchSize)])
pStart['s'] = matrix([1.0]*(2*batchSize))
###############################################################################


def read_data(data_iter, batch_id):
    data = data_iter.next()
    batch_id += 1
    real_cpu, _ = data
    real_data = real_cpu.clone().to(device)
    real.resize_as_(real_data).copy_(real_data)
    noise.resize_(batchSize, nz, 1, 1).normal_(0, 1)
    with torch.no_grad():
        fake = netG(noise).detach()

    return real, fake, real_cpu, noise, batch_id


def comput_dist(real, fake):
    num_r = real.size(0)
    num_f = fake.size(0)
    real_flat = real.view(num_r, -1)
    fake_flat = fake.view(num_f, -1)

    real3D = real_flat.unsqueeze(1).expand(num_r, num_f, data_dim)
    fake3D = fake_flat.unsqueeze(0).expand(num_r, num_f, data_dim)
    # compute squared L2 distance
    dif = real3D - fake3D
    dist = 0.5 * dif.pow(2).sum(2).squeeze()

    return dist


def Wasserstein_LP(dist):
    b = matrix(dist.cpu().double().numpy().flatten())
    sol = solvers.lp(c, A, b, primalstart=pStart, solver='glpk')
    offset = 0.5 * (sum(sol['x'])) / batchSize
    sol['x'] = sol['x'] - offset
    pStart['x'] = sol['x']
    pStart['s'] = sol['s']

    return sol


def approx_OT(sol):
    ###########################################################################
    ################ Compute the OT mapping for each fake data ################
    # ResMat = np.array(sol['s']).reshape((batchSize,batchSize))
    # mapping = torch.from_numpy(np.argmin(ResMat, axis=0)).long().to(device)

    ResMat = np.array(sol['z']).reshape((batchSize, batchSize))
    mapping = torch.from_numpy(np.argmax(ResMat, axis=0)).long().to(device)
    
    return mapping
    ###########################################################################


###############################################################################
################## Optimal Transport Regularization ###########################
###############################################################################
## f(y) = inf { f(x) + c(x,y) }
## 0 \in grad_x { f(x) + c(x,y) }
###############################################################################
def OT_regularization(output_fake, fake, RF_dif):
    output_fake_grad = torch.ones(output_fake.size()).to(device)
    gradients = autograd.grad(outputs=output_fake, inputs=fake,
                              grad_outputs=output_fake_grad,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    n = gradients.size(0)    
    RegLoss = 0.5 * ((gradients.view(n, -1).norm(dim=1) / (2*Kr) - Kr/2 * RF_dif.view(n, -1).norm(dim=1)).pow(2)).mean()
    fake.requires_grad = False

    return RegLoss


def update_average(model_tgt, model_src, beta):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.data.copy_(beta*p_tgt.data + (1. - beta)*p_src.data)

            
def load_fixed_noise(img_path, device):
    file_name = '{0}/fixed_noise.pth'.format(img_path)
    if os.path.isfile(file_name):
        fixed_noise = torch.load(file_name, map_location=torch.device('cpu'))
    else:
        fixed_noise = torch.FloatTensor(genImg_num, nz, 1, 1).normal_(0, 1)
        torch.save(fixed_noise, file_name)

    return fixed_noise.to(device)


def save_checkpoint(checkpoint_path, epoch, Giter):
    torch.save(netG_test.state_dict(), '{0}/netG_epoch_{1}_Giter_{2}.pth'.format(checkpoint_path, epoch, Giter))
    torch.save(netD.state_dict(), '{0}/netD_epoch_{1}_Giter_{2}.pth'.format(checkpoint_path, epoch, Giter))


def save_images(img_path, real_cpu):
    real_cpu = real_cpu[:genImg_num].mul(0.5).add(0.5)
    vutils.save_image(real_cpu, '{0}/real_samples.png'.format(img_path), nrow=genImg_num_sroot)
    netG_test.eval()
    with torch.no_grad():
        fake = netG_test(fixed_noise)
    fake.data = fake.cpu().data.mul(0.5).add(0.5)
    vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(img_path, Giter), nrow=genImg_num_sroot)


fixed_noise = load_fixed_noise(img_path, device)
num_batches = len(dataloader)
data_iter = iter(dataloader)
batch_id = 0
optimizerD, optimizerG = build_optimizerD(Optimizer), build_optimizerG(Optimizer)
schedulerD = build_lr_scheduler(optimizerD, milestones, lr_anneal)
schedulerG = build_lr_scheduler(optimizerG, milestones, lr_anneal)
epoch, Giter = load_last_checkpoint(netD, netG, checkpoint_path)
WD = torch.FloatTensor(1)

if lr_anneal != 1.0:
    for i in range(Giter-1):
        schedulerD.step()
        schedulerG.step()

netG_copied = False
if EMA > 0 and Giter >= EMA_startIter:
    netG_test = deepcopy(netG)
    netG_copied = True
else:
    netG_test = netG

while epoch <= epochs and Giter <= Giters:
    ###########################################################################
    #               (1) Update the Discriminator networks D
    ###########################################################################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    netD.train()
    netG.train()
    ###########################################################################
    #                    Deep Regression for discriminator
    ###########################################################################
    ##################### perform deep regression for D #######################
    j = 0
    while j < Diters:
        j += 1
        if batch_id >= num_batches:
            data_iter = iter(dataloader)
            batch_id = 0
            if epoch % save_ckpt_epoch_freq == 0:
                save_checkpoint(checkpoint_path, epoch, Giter)
            epoch += 1

        real, fake, real_cpu, noise, batch_id = read_data(data_iter, batch_id)

        dist = comput_dist(real, fake)
        dist = K * dist
        sol = Wasserstein_LP(dist)
        if LAMBDA > 0:
            mapping = approx_OT(sol)
            real_ordered = real[mapping]  # match real and fake
            RF_dif = real_ordered - fake

        # construct target
        target = torch.from_numpy(np.array(sol['x'])).float()
        target = target.squeeze().to(device)

        for k in range(DOptIters):
            netD.zero_grad()
            fake.requires_grad_()
            if fake.grad is not None:
                fake.grad.data.zero_()
            output_real = netD(real)
            output_fake = netD(fake)
            output_real, output_fake = output_real.squeeze(), output_fake.squeeze()
            output_R_mean = output_real.mean(0).view(1)
            output_F_mean = output_fake.mean(0).view(1)

            L2LossD_real = criterion(output_R_mean[0], target[:batchSize].mean())
            L2LossD_fake_1 = criterion(output_F_mean[0], target[batchSize:].mean())
            L2LossD_fake_2 = criterion(output_fake, target[batchSize:])
            L2LossD_fake = alpha * L2LossD_fake_1 + (1 - alpha) * L2LossD_fake_2
            L2LossD = 0.5 * L2LossD_real + 0.5 * L2LossD_fake 

            if LAMBDA > 0:
                RegLossD = OT_regularization(output_fake, fake, RF_dif)
                TotalLoss = L2LossD + LAMBDA * RegLossD
            else:
                TotalLoss = L2LossD

            TotalLoss.backward()
            optimizerD.step()

        WD = output_R_mean - output_F_mean  # Wasserstein Distance

    #################### Discriminator Regression done ########################

    ###########################################################################
    #                   (2) Update the Generator network G
    ###########################################################################
    for p in netD.parameters():
        p.requires_grad = False  # frozen D
    ###########################################################################
    ##                               Update G
    ###########################################################################

    netG.zero_grad()
    fake = netG(noise)
    output_fake = netD(fake)
    output_F_mean_after = output_fake.mean(0).view(1)
    output_F_mean_after.backward(mone)
    optimizerG.step()

    schedulerD.step()
    schedulerG.step()

    if EMA > 0:
        if not netG_copied and Giter >= EMA_startIter:
            netG_test = deepcopy(netG)
            netG_copied = True
        if Giter >= EMA_startIter:
            update_average(netG_test, netG, EMA)
    
    Giter += 1

    G_growth = output_F_mean_after - output_F_mean

    if Giter % verbose_freq == 0:
        if LAMBDA > 0:
            log_str = '[{:d}/{:d}][{:d}] | WD {:.9f} | real_mean {:.9f} | fake_mean {:.9f} | G_growth {:.9f} | ' \
                      'L2LossD_real {:.9f} | L2LossD_fake {:.9f} | ' \
                      'L2LossD {:.9f} | RegLossD {:.9f} | TotalLoss {:.9f}'.format(
                epoch, epochs, Giter,
                WD.item(), output_R_mean.item(), output_F_mean.item(), G_growth.item(),
                L2LossD_real.item(), L2LossD_fake.item(),
                L2LossD.item(), RegLossD.item(), TotalLoss.item())
        else:
            log_str = '[{:d}/{:d}][{:d}] | WD {:.9f} | real_mean {:.9f} | fake_mean {:.9f} | G_growth {:.9f} | ' \
                      'L2LossD_real {:.9f} | L2LossD_fake {:.9f} | ' \
                      'L2LossD {:.9f} | TotalLoss {:.9f}'.format(
                epoch, epochs, Giter,
                WD.item(), output_R_mean.item(), output_F_mean.item(), G_growth.item(),
                L2LossD_real.item(), L2LossD_fake.item(),
                L2LossD.item(), TotalLoss.item())
        print(log_str)
        with open(log_fileName, 'a') as f:
            f.write('{}\n'.format(log_str))

    if Giter % genImg_freq == 0:
        save_images(img_path, real_cpu)

    if Giter % save_ckpt_freq == 0:
        save_checkpoint(checkpoint_path, epoch, Giter)


