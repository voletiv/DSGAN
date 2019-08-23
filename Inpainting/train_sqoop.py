import argparse
import datetime
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
from cv2 import putText

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torch
# from tensorboardX import SummaryWriter

from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
# from torch.autograd import Variable

from datasets_mask import *
from models import *
from models.layer_util import *
from generate_my_sqoop import *


def mem_check():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    print("Mem:", process.memory_info().rss/1024/1024/1024, "GB")


def plot_losses(save_path, iters, g_losses, g_loss_labels, d_losses, d_loss_labels):
    fig = plt.figure(figsize=(20, 20))
    plt.subplot(211)
    plt.plot(iters, np.zeros(len(iters)), 'k--', alpha=0.5)
    for i in range(len(g_losses)):
        plt.plot(iters, g_losses[i], label=g_loss_labels[i])
    plt.legend()
    # plt.yscale("symlog")
    plt.title("Generator loss")
    plt.xlabel("Epochs")
    plt.subplot(212)
    plt.plot(iters, np.zeros(len(iters)), 'k--', alpha=0.5)
    for i in range(len(d_losses)):
        plt.plot(iters, d_losses[i], label=d_loss_labels[i])
    plt.legend()
    # plt.yscale("symlog")
    plt.title("Discriminator loss")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(save_path, "plots.png"), bbox_inches='tight', pad_inches=0.5)
    plt.clf()
    plt.close()


parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--cond_mode', type=str, choices=['obj', 'obj_obj', 'obj_rel_obj'], help='structure of condition')
parser.add_argument('--save_path', type=str, default='.', help='location to save path')
parser.add_argument('--num_objects', type=int, default=5)
parser.add_argument('--pairings_per_obj', type=int, default=0)
parser.add_argument('--num_repeats', type=int, default=10)
parser.add_argument('--val', action='store_true')
parser.add_argument('--num_repeats_eval', type=int, default=10)
# Training
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')
parser.add_argument('--name', type=str, default='SQOOP', help='name of the experiment')
# Model
# parser.add_argument('--crop_mode', type=str, default='trans_ctr', help='[random | trans_ctr]')
parser.add_argument('--cond_dim', type=int, default=128, help='dimmension for each element in condition')
parser.add_argument('--noise_dim', type=int, default=128, help='dimmension for noise vector')
parser.add_argument('--ngf', type=int, default=64, help='dimensionality of the latent space in G')
parser.add_argument('--n_layers_G', type=int, default=3, help='number of layers in generator')
parser.add_argument('--n_blocks_G', type=int, default=3, help='number of blocks in generator')
parser.add_argument('--n_layers_D', type=int, default=3, help='number of layers in discriminator')
parser.add_argument('--num_D', type=int, default=2, help='number of discriminators for multiscale PatchGAN')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
# Image
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
parser.add_argument('--img_nc', type=int, default=1, help='# of channels in image')
# parser.add_argument('--load_size', type=int, default=280, help='size of each image loading dimension')
# parser.add_argument('--mask_size', type=int, default=128, help='size of random mask')
# parser.add_argument('--channels', type=int, default=3, help='number of image channels')
# Steps
parser.add_argument('--log_interval', type=int, default=500, help='interval (in iters) between image sampling')
parser.add_argument('--sample_interval', type=int, default=500, help='interval (in iters) between image sampling')
parser.add_argument('--snapshot_interval', type=int, default=5, help='interval (in epochs) between model saving')
# Diversity
parser.add_argument('--feat_w', type=float, default=10, help='weights for diversity-encouraging term')
parser.add_argument('--noise_w', type=float, default=5, help='weights for diversity-encouraging term')
# parser.add_argument('--no_noise', action='store_true', help='do not use the noise if specified')
parser.add_argument('--dist_measure', type=str, default='perceptual', help='[rgb | perceptual]')
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()
device = 'cuda:' + str(opt.gpu_id) if cuda else 'cpu'

# visualization
exp_name = '{:%Y%m%d_%H%M%S}_{}_SQOOP_objs{}perObj{}reps{}_cond{}_z{}_ngf{}_G{}numG{}_D{}numD{}_lr{:.4f}_featW{:.3f}_noiseW{:.3f}_{}'.format(datetime.datetime.now(),
    opt.cond_mode, opt.num_objects, opt.pairings_per_obj, opt.num_repeats, opt.cond_dim, opt.noise_dim, opt.ngf,
    opt.n_layers_G, opt.n_blocks_G, opt.n_layers_D, opt.num_D, opt.lr, opt.feat_w, opt.noise_w, opt.dist_measure)
exp_name += '_' + opt.name

# if not opt.crop_mode == 'random':
#     exp_name += '_' + opt.crop_mode
checkpoint_dir = os.path.join(opt.save_path, exp_name, 'weights')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

samples_dir = os.path.join(opt.save_path, exp_name, 'samples')
os.makedirs(samples_dir)

# writer = SummaryWriter(log_dir=checkpoint_dir)

# Loss function
# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
from models.my_losses import GANLoss
criterionGAN = GANLoss(use_lsgan=True, device=device)
criterionFeat = torch.nn.L1Loss()

# Initialize generator
from models.my_Generator_NET import Generator
generator = Generator(cond_mode=opt.cond_mode, cond_dim=opt.cond_dim, z_dim=opt.noise_dim, ngf=opt.ngf,
                        n_upsampling=opt.n_layers_G, n_blocks=opt.n_blocks_G, output_nc=opt.img_nc)

# Initialize discriminator
from models.Discriminator_NET import MultiscaleDiscriminator
discriminator = MultiscaleDiscriminator(input_nc=opt.img_nc, n_layers=opt.n_layers_D, num_D=opt.num_D)

if cuda:
    torch.cuda.set_device(opt.gpu_id)
    generator.cuda()
    discriminator.cuda()
    criterionGAN.cuda()
    criterionFeat.cuda()

# Initialize weights
generator.apply(weights_init)
discriminator.apply(weights_init)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# # Dataset loader
# batch_size_test = opt.batch_size
# transforms_ = [ transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
# dataloader = DataLoader(ImageDataset("./data/%s" % opt.dataset_name, \
#                                      img_size=opt.img_size, load_size=opt.load_size, mask_size=opt.mask_size, crop_mode=opt.crop_mode),
#                         batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
# test_dataloader = DataLoader(ImageDataset("./data/%s" % opt.dataset_name, \
#                                           img_size=opt.img_size, load_size=opt.load_size, mask_size=opt.mask_size, mode='val'),
#                         batch_size=batch_size_test, shuffle=True, num_workers=1)

# Data

data = gen_my_sqoop(num_objects=opt.num_objects, pairings_per_obj=opt.pairings_per_obj, num_repeats=opt.num_repeats, val=opt.val, num_repeats_eval=opt.num_repeats_eval)
if opt.val:
    train_imgs, train_qs, train_qs_oh, _, val_imgs, val_qs, val_qs_oh, _ = data
else:
    train_imgs, train_qs, train_qs_oh, _ = data

# ims : consider only green channel, and make it BxCxWxH
train_imgs = torch.from_numpy(np.array(train_imgs)[:, :, :, 1, np.newaxis]).permute(0, 3, 1, 2).float().div(128.)
# cond : in the format obj_rel_obj
train_cond = torch.from_numpy(np.array(train_qs_oh))

# import pdb; pdb.set_trace()

if opt.cond_mode == 'obj':
    train_cond = train_cond[:, 0]
elif opt.cond_mode == 'obj_obj':
    train_cond = train_cond[:, [0, 2]]
elif opt.cond_mode == 'obj_rel_obj':
    pass

train_dataset = TensorDataset(train_imgs, train_cond)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, pin_memory=True)

if opt.val:
    val_imgs = torch.from_numpy(np.array(val_imgs)[:, :, :, 1, np.newaxis]).permute(0, 3, 1, 2)
    val_cond = torch.from_numpy(np.array(val_qs_oh))
    if opt.cond_mode == 'obj':
        val_cond = val_cond[:, 0]
    elif opt.cond_mode == 'obj_obj':
        val_cond = val_cond[:, [0, 2]]
    elif opt.cond_mode == 'obj_rel_obj':
        pass
    val_dataset = TensorDataset(val_imgs, val_cond)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size_test, shuffle=False, num_workers=opt.n_cpu, pin_memory=True)

# Fixed data
fixed_noise = torch.randn(opt.batch_size, opt.noise_dim)
if opt.val:
    fixed_idx = np.random.choice(len(val_cond), opt.batch_size)
    fixed_cond = val_cond[fixed_idx]
    fixed_qs = [' '.join(val_qs[i]) for i in fixed_idx]
else:
    fixed_idx = np.random.choice(len(train_cond), opt.batch_size)
    fixed_cond = train_cond[fixed_idx]
    fixed_qs = [' '.join(train_qs[i]) for i in fixed_idx]

fixed_cond = fixed_cond.to(device)
fixed_noise = fixed_noise.to(device)


# def sample_test(test_num_noise=8):
#     """
#     Run inapinting
#     Inputs:
#         test_num_noise: number of latent codes to visualize
#     Outputs:
#         gen_mask: generated images
#     """
#     samples, masked_samples, i, masks = next(iter(test_dataloader))
#     samples = Variable(samples.type(Tensor))
#     masked_samples = Variable(masked_samples.type(Tensor))
#     masks = Variable(masks.type(Tensor))

#     # pad the gt for visualization
#     i = i.repeat(test_num_noise)
#     paded_samples = masked_samples.repeat(test_num_noise, 1, 1, 1)
#     padded_masks = masks.repeat(test_num_noise, 1, 1, 1)
#     noise = torch.randn(samples.size(0)*test_num_noise, opt.noise_dim, 1, 1)
#     noise = Variable(noise.cuda())
#     # Generate inpainted image
#     generator.eval()
#     gen_mask = generator(paded_samples, noise, padded_masks)
#     generator.train()

#     return gen_mask


# Log file
log_file_name = os.path.join(exp_name, 'log.txt')
log_file = open(log_file_name, "wt")

# -----------------
#  Run Training
# -----------------

ckpts = []
d_losses = []
d_real_losses = []
d_fake_losses = []
g_losses = []
g_adv_losses = []
g_feat_losses = []
g_noise_losses = []
# d_xs = []
# d_gz_ds = []
# d_gz_gs = []

try:
    start_time = time.time()
    for epoch in range(opt.n_epochs):
        for batch, (imgs, conds) in enumerate(train_dataloader):

            # print("Epoch", epoch, "; batch", batch, "of", len(train_dataloader)-1)
            # mem_check()
            if batch < 22:
                print("\n\nBatch", batch)
                subprocess.run('nvidia-smi')

            # Configure input
            imgs = imgs.to(device)
            conds = conds.to(device)

            # double the batches
            imgs = torch.cat((imgs, imgs),dim=0)
            conds = torch.cat((conds, conds), dim=0)

            # sample noises
            B = int(imgs.size(0)/2)
            noise = torch.randn(B*2, opt.noise_dim).to(device)

            # run generation
            gen_imgs = generator(conds, noise)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            pred_real = discriminator(imgs)
            pred_fake = discriminator(gen_imgs.detach())
            real_loss = criterionGAN(pred_real, True)
            fake_loss = criterionGAN(pred_fake, False)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            # Adversarial loss
            pred_fake = discriminator(gen_imgs)
            g_adv = criterionGAN(pred_fake, True)

            # Pixelwise loss
            g_feat = 0
            feat_weights = 4.0 / (opt.n_layers_D + 1)
            D_weights = 1.0 / opt.num_D
            for i in range(opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    g_feat += D_weights * feat_weights * \
                        criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * opt.feat_w

            # noise sensitivity loss
            if opt.dist_measure == 'rgb':
                g_noise_out_dist = torch.mean(torch.abs(gen_parts[:B] - gen_parts[B:]))
            elif opt.dist_measure == 'perceptual':
                g_noise_out_dist = 0
                for i in range(opt.num_D):
                    for j in range(len(pred_fake[i])-1):
                        g_noise_out_dist += D_weights * feat_weights * \
                            torch.mean(torch.abs(pred_fake[i][j][:B] - pred_fake[i][j][B:]).view(B,-1),dim=1)

            g_noise_z_dist = torch.mean(torch.abs(noise[:B] - noise[B:]).view(B,-1), dim=1)
            g_noise = torch.mean( g_noise_out_dist / g_noise_z_dist ) * opt.noise_w

            # Total loss
            g_loss = g_adv + g_feat - g_noise
            g_loss.backward()
            optimizer_G.step()

            # Generate sample at sample interval
            batches_done = epoch * len(train_dataloader) + batch
            if batches_done % opt.log_interval == 0:

                # writer.add_scalar('d_loss', d_loss.item(), batches_done/len(train_dataloader))
                # writer.add_scalar('g_loss', g_loss.item(), batches_done/len(train_dataloader))
                # writer.add_scalar('g_adv', g_adv.item(), batches_done/len(train_dataloader))
                # writer.add_scalar('g_feat', g_feat.item(), batches_done/len(train_dataloader))
                # writer.add_scalar('g_noise', g_noise.item(), batches_done/len(train_dataloader))

                ckpts.append(batches_done/len(train_dataloader))
                d_losses.append(d_loss.item())
                d_real_losses.append(real_loss.item()/2)
                d_fake_losses.append(fake_loss.item()/2)
                g_losses.append(g_loss.item())
                g_adv_losses.append(g_adv.item())
                g_feat_losses.append(g_feat.item())
                g_noise_losses.append(g_noise.item())
                # d_xs.append(pred_real.mean().item())
                # d_gz_ds.append(pred_fake.mean().item())
                # d_gz_gs.append(pred_fake_g.mean().item())

                curr_time = time.time()
                curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
                elapsed = str(datetime.timedelta(seconds=(curr_time - start_time)))

                # log = "[%s] [Elapsed %s] [Epoch %d/%d] [Batch %d/%d] [D loss: %.4f, real: %.4f, fake: %.4f] [G loss: %.4f, adv: %.4f, pixel: %.4f, noise: %.4f] [D(x): %.4f, D(G(z))_D: %.4f, D(G(z))_G: %.4f]" % (
                log = "[%s] [Elapsed %s] [Epoch %d/%d] [Batch %d/%d] [D loss: %.4f, real: %.4f, fake: %.4f] [G loss: %.4f, adv: %.4f, pixel: %.4f, noise: %.4f]" % (
                        curr_time_str, elapsed, epoch, opt.n_epochs, batch, len(train_dataloader),
                        d_losses[-1], d_real_losses[-1], d_fake_losses[-1], g_losses[-1], g_adv_losses[-1], g_feat_losses[-1], g_noise_losses[-1])
                        # d_xs[-1], d_gz_ds[-1], d_gz_gs[-1])
                log += "\n"
                print(log)
                log_file.write(log)
                log_file.flush()

                # plot_losses(ckpts, d_losses, g_losses, g_adv_losses, g_feat_losses, g_noise_losses, d_xs, d_gz_ds, d_gz_gs)
                plot_losses(os.path.join(opt.save_path, exp_name), ckpts,
                            [g_losses, g_adv_losses, g_feat_losses, g_noise_losses], ["G_loss", "G_loss_adv", "G_loss_feat", "G_loss_noise"],
                            [d_losses, d_real_losses, d_fake_losses], ["D_loss", "D_loss_real", "D_loss_fake"])

            if batches_done % opt.sample_interval == 0:

                generator.eval()
                gen_imgs = generator(fixed_cond, fixed_noise)
                generator.train()

                # record for visualization
                gen_imgs = gen_imgs.detach().cpu().clamp_(0, 1)
                gen_imgs = torch.cat([torch.zeros(gen_imgs.shape), gen_imgs, torch.zeros(gen_imgs.shape)], dim=1)
                # Add text of condition to image
                sample_images_bhwc = gen_imgs.permute(0, 2, 3, 1).numpy()
                ims = np.empty((0, sample_images_bhwc.shape[1]+10, *sample_images_bhwc.shape[2:]))
                for i in range(len(sample_images_bhwc)):
                    im = sample_images_bhwc[i]
                    line = fixed_qs[i]
                    im = putText(np.concatenate((im, np.ones((10, im.shape[1], im.shape[2]))), axis=0),
                                     line, (5, im.shape[0]+7), 0, .3, (0, 0, 0), 1)
                    ims = np.vstack((ims, im[np.newaxis, :, :, :]))
                gen_imgs = torch.from_numpy(ims).permute(0, 3, 1, 2)

                save_image(gen_imgs, os.path.join(samples_dir, f'vis_{batches_done:06d}.png'), padding=4, nrow=10, pad_value=1)
                # writer.add_image('gen_imgs', train_gen_imgs, batches_done/len(dataloader))
                #test_img = save_sample(batches_done)

                # # save the latest network
                # save_path = os.path.join(checkpoint_dir, 'netG_latest.pth')
                # torch.save(generator.state_dict(), save_path)

            # if batches_done % opt.test_sample_interval == 0:
            #     test_gen_imgs = sample_test(8)
            #     test_gen_imgs = make_grid(test_gen_imgs.cpu().data, normalize=True, scale_each=True, nrow=batch_size_test)
            #     writer.add_image('pred_patch_test', test_gen_imgs, batches_done/len(dataloader))

        if (epoch+1) % opt.snapshot_interval == 0:
            torch.save({
                        'epoch': epoch,
                        'G_state_dict': generator.module.state_dict() if hasattr(generator, "module") else generator.state_dict(),    # "module" in case DataParallel is used
                        'G_optimizer_state_dict': optimizer_G.state_dict(),
                        'D_state_dict': discriminator.module.state_dict() if hasattr(discriminator, "module") else discriminator.state_dict(),    # "module" in case DataParallel is used,
                        'D_optimizer_state_dict': optimizer_D.state_dict(),
                        }, os.path.join(checkpoint_dir, f'ckpt_{epoch:07d}.pth'))

except KeyboardInterrupt:
    print("Ctrl+C pressed!")

torch.save({
            'epoch': epoch,
            'G_state_dict': generator.module.state_dict() if hasattr(generator, "module") else generator.state_dict(),    # "module" in case DataParallel is used
            'G_optimizer_state_dict': optimizer_G.state_dict(),
            'D_state_dict': discriminator.module.state_dict() if hasattr(discriminator, "module") else discriminator.state_dict(),    # "module" in case DataParallel is used,
            'D_optimizer_state_dict': optimizer_D.state_dict(),
            }, os.path.join(checkpoint_dir, f'final_model_state_dict_epoch{epoch:07d}.pth'))

torch.save({
            'epoch': epoch,
            'G_state_dict': generator.module if hasattr(generator, "module") else generator,    # "module" in case DataParallel is used
            'G_optimizer_state_dict': optimizer_G,
            'D_state_dict': discriminator.module if hasattr(discriminator, "module") else discriminator,    # "module" in case DataParallel is used,
            'D_optimizer_state_dict': optimizer_D,
            }, os.path.join(checkpoint_dir, f'final_model_epoch{epoch:07d}.pth'))

