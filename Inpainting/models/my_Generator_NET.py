import numpy as np
import functools
import torch
import torch.nn as nn

from torch.autograd import Variable

from .layer_util import *


def get_norm_layer1d(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm1d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class Generator(nn.Module):
    def __init__(self, cond_mode='obj_rel_obj', cond_dim=128, z_dim=128, ngf=64, 
                 n_upsampling=3, n_blocks=9, norm_layer='instance', output_imsize=64,
                 padding_type='reflect', use_skip=False, use_mask=False, extra_embed=False,
                 obj_vocab_len=36, rel_vocab_len=4, output_nc=1):

        assert cond_mode in ['obj', 'obj_obj', 'obj_rel_obj']
        assert(n_blocks >= 0)

        super(Generator, self).__init__()

        self.cond_mode = cond_mode
        self.cond_dim = cond_dim
        self.z_dim = z_dim
        self.output_imsize = output_imsize
        self.ngf = ngf
        self.n_upsampling = n_upsampling
        self.n_blocks = n_blocks
        self.norm_layer1d = get_norm_layer1d(norm_layer)
        self.norm_layer = get_norm_layer(norm_layer)
        self.padding_type = padding_type
        self.use_skip = use_skip
        self.use_mask=use_mask
        self.extra_embed = extra_embed
        self.obj_vocab_len = obj_vocab_len
        self.rel_vocab_len = rel_vocab_len
        self.output_nc = output_nc

        self.activation = nn.ReLU(True)
        self.latent_size = self.output_imsize // 2**n_upsampling
        self.feat_dim = self.ngf * 2**n_upsampling

        # ctx_dim = 3 if not extra_embed else 6
        # self.ctx_input_embedder = self.get_input(ctx_dim)
        # self.ctx_downsampler = self.get_downsampler()
        self.obj_embedder = nn.Embedding(self.obj_vocab_len, self.cond_dim)

        if self.cond_mode == 'obj_rel_obj':
            self.rel_embedder = nn.Embedding(self.rel_vocab_len, self.cond_dim)
            self.cond_emb_dim = self.cond_dim*3 + self.z_dim
        elif self.cond_mode == 'obj_obj':
            self.cond_emb_dim = self.cond_dim*2 + self.z_dim
        elif self.cond_mode == 'obj':
            self.cond_emb_dim = self.cond_dim + self.z_dim

        # import pdb; pdb.set_trace()
        self.noise_fuser = self.fuse_layer()
        self.latent_embedder = self.res_blocks(self.feat_dim, self.n_blocks)
        self.decoder = self.upsampler()
        self.output_layer = self.get_output()

    # def get_input(self, input_nc):
    #     model = [nn.ReflectionPad2d(3),
    #             nn.Conv2d(input_nc, self.ngf, kernel_size=7, padding=0),
    #             self.norm_layer(self.ngf),
    #             self.activation]
    #     return nn.Sequential(*model)

    # def get_downsampler(self):
    #     ### downsample
    #     model = []
    #     for i in range(self.n_upsampling):
    #         mult = 2**i
    #         model += [nn.Conv2d(self.ngf * mult, self.ngf * mult * 2, kernel_size=3, stride=2, padding=1),
    #                   self.norm_layer(self.ngf * mult * 2),
    #                   self.activation]
    #     return nn.Sequential(*model)

    # def forward_encoder(self, input_embedder, encoder, input, use_skip):
    #     enc_feats = []
    #     enc_feat = input_embedder(input)
    #     for i, layer in enumerate(encoder):
    #         enc_feat = layer(enc_feat)
    #         if use_skip and ((i < self.n_upsampling*3-1) and (i % 3 == 2)): # super-duper hard-coded
    #             enc_feats.append(enc_feat)
    #     return enc_feat, enc_feats

    # def fuse_layer(self):
    #     model = [nn.Conv2d(self.cond_dim*3 + self.z_dim, self.feat_dim, kernel_size=3, padding=1),
    #             self.norm_layer(self.feat_dim),
    #             self.activation]
    #     return nn.Sequential(*model)

    def fuse_layer(self):
        model = [nn.Linear(self.cond_emb_dim, self.feat_dim*self.latent_size*self.latent_size),
                self.norm_layer1d(self.feat_dim*self.latent_size*self.latent_size),
                self.activation]
        return nn.Sequential(*model)

    def res_blocks(self, feat_dim, n_blocks):
        ### resnet blocks
        model = []
        for i in range(n_blocks):
            model += [ResnetBlock(feat_dim,
                padding_type=self.padding_type,
                activation=self.activation,
                norm_layer=self.norm_layer)]
        return nn.Sequential(*model)

    def upsampler(self):
        ### upsample
        model = []
        for i in range(self.n_upsampling):
            mult = 2**(self.n_upsampling - i)
            dim_in = self.ngf * mult
            dim_out = int(self.ngf * mult / 2)
            if self.use_skip and i > 0:
                dim_in = dim_in*2
            model += [nn.ConvTranspose2d(dim_in, dim_out,
                            kernel_size=3, stride=2, padding=1, output_padding=1),
                       self.norm_layer(int(self.ngf * mult / 2)),
                       self.activation]
        return nn.Sequential(*model)

    def get_output(self):
        model = [nn.ReflectionPad2d(3),
                nn.Conv2d(self.ngf, self.output_nc, kernel_size=7, padding=0),
                nn.Tanh()]
        return nn.Sequential(*model)

    def forward_decoder(self, dec_feat):
        for i, layer in enumerate(self.decoder):
            # if (self.use_skip and len(enc_feats) > 0) and ((i > 0) and (i % 3 ==0)): # super-duper hard-coded
            #     dec_feat = torch.cat((enc_feats[-int((i-3)/3)-1], dec_feat),1)
            dec_feat = layer(dec_feat)
        output = self.output_layer(dec_feat)
        return output

    def forward(self, cond, noise, mask=None):
        # ctx_feat, ctx_feats = self.forward_encoder(self.ctx_inputEmbedder, self.ctx_downsampler, img, self.use_skip)
        if self.cond_mode == 'obj_rel_obj':
            obj1, rel, obj2 = cond[:, 0], cond[:, 1], cond[:, 2]
            obj1_emb, rel_emb, obj2_emb = self.obj_embedder(obj1), self.rel_embedder(rel), self.obj_embedder(obj2)
            cond_emb = torch.cat([obj1_emb, rel_emb, obj2_emb], dim=1)
        elif self.cond_mode == 'obj_obj':
            obj1, obj2 = cond[:, 0], cond[:, 1]
            obj1_emb, obj2_emb = self.obj_embedder(obj1), self.obj_embedder(obj2)
            cond_emb = torch.cat([obj1_emb, obj2_emb], dim=1)
        elif self.cond_mode == 'obj':
            cond_emb = self.obj_embedder(cond)

        # fuse the noise with feature
        combined_feat = torch.cat([cond_emb, noise], dim=1)
        # import pdb; pdb.set_trace()
        combined_feat = self.noise_fuser(combined_feat)
        combined_feat = combined_feat.view(-1, self.feat_dim, self.latent_size, self.latent_size)
        # pass thru res blocks
        embed_feat = self.latent_embedder(combined_feat)
        # output
        output = self.forward_decoder(embed_feat)
        if self.use_mask:
            mask_output = mask.repeat(1, self.output_nc, 1, 1)
            output = (1 - mask_output)*img[:,:3,:,:] + mask_output*output

        return output
