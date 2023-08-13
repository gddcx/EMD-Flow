import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import BasicLayer
from corr import CorrBlock
from utils.utils import coords_grid

from cfp import CFP


class OFE(nn.Module):
    def __init__(self, args):
        super(OFE, self).__init__()
        dim = args.m_dim
        mlp_scale = 4
        window_size = [8, 8]
        num_layers = [2, 2]

        self.num_layers = len(num_layers)
        self.blocks = nn.ModuleList()
        for n in range(self.num_layers):
            if n == self.num_layers - 1:
                self.blocks.append(
                    BasicLayer(num_layer=num_layers[n], dim=dim, mlp_scale=mlp_scale, window_size=window_size, cross=False))
            else:
                self.blocks.append(
                    BasicLayer(num_layer=num_layers[n], dim=dim, mlp_scale=mlp_scale, window_size=window_size, cross=True))

    def forward(self, fmap1, fmap2):
        _, _, ht, wd = fmap1.shape
        pad_h, pad_w = (8 - (ht % 8)) % 8, (8 - (wd % 8)) % 8
        _pad = [pad_w // 2, pad_w - pad_w // 2, pad_h, 0]
        fmap1 = F.pad(fmap1, pad=_pad, mode='constant', value=0)
        fmap2 = F.pad(fmap2, pad=_pad, mode='constant', value=0)
        mask = torch.zeros([1, ht, wd]).to(fmap1.device)
        mask = torch.nn.functional.pad(mask, pad=_pad, mode='constant', value=1)
        mask = mask.bool()
        fmap1 = fmap1.permute(0, 2, 3, 1).contiguous().float()
        fmap2 = fmap2.permute(0, 2, 3, 1).contiguous().float()

        for idx, blk in enumerate(self.blocks):
            fmap1, fmap2 = blk(fmap1, fmap2, mask=mask)

        _, ht, wd, _ = fmap1.shape
        fmap1 = fmap1[:, _pad[2]:ht - _pad[3], _pad[0]:wd - _pad[1], :]
        fmap2 = fmap2[:, _pad[2]:ht - _pad[3], _pad[0]:wd - _pad[1], :]

        fmap1 = fmap1.permute(0, 3, 1, 2).contiguous()
        fmap2 = fmap2.permute(0, 3, 1, 2).contiguous()

        return fmap1, fmap2


class MMA(nn.Module):
    def __init__(self, args):
        super(MMA, self).__init__()
        self.ofe = OFE(args)
        self.cfp = CFP(c_dim=args.c_dim)

        self.args = args
        self.multi_scale = True
        if self.multi_scale:
            chnn_hid = 32
            self.level_corr = 2
            self.gamma = nn.Parameter(torch.zeros(1))
            dila_s = [4, 8, 16]
            conv_s = [nn.Sequential(
                nn.Conv2d(2 * self.level_corr, chnn_hid, 3, dilation=ii, padding=ii),
                nn.ReLU(inplace=True))
                for ii in dila_s]
            self.conv_s = nn.ModuleList(conv_s)
            chnn_ic = 256
            self.conv_rd = nn.Sequential(
                nn.Conv2d(len(dila_s) * chnn_hid, chnn_ic, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(chnn_ic, 2, 3, 1, 1))
            print(' -- Using multi-scale correlations for init_flow --')
            print(f' -- Number of Scale: {self.level_corr} --')

    def forward(self, fmap1, fmap2, inp):
        batch, ch, ht, wd = fmap1.shape
        fmap1, fmap2 = self.ofe(fmap1, fmap2)
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)

        corr_i = corr_fn.corr_pyramid[0]
        h_d, w_d = corr_i.shape[-2:]
        corr_sm = torch.softmax(corr_i.view(batch, ht * wd, h_d * w_d), dim=-1)

        crds_d = coords_grid(batch, h_d, w_d, device=corr_sm.device).reshape(batch, h_d * w_d, 2)
        crds = coords_grid(batch, ht, wd, device=corr_sm.device).reshape(batch, ht * wd, 2)
        flo = (corr_sm @ crds_d) * (ht / h_d) - crds

        flow_attn, conf, self_corr = self.cfp(inp=inp, corr_sm=corr_sm)

        flo = conf * flo + (1 - conf) * (flow_attn @ flo)
        flo_0 = flo.reshape(batch, ht, wd, 2).permute(0, 3, 1, 2).contiguous()

        if self.multi_scale:
            flo_s = []
            for ii in range(self.level_corr):
                corr_i = torch.nn.functional.avg_pool2d(corr_i, 2, stride=2)
                h_d, w_d = corr_i.shape[-2:]
                corr_sm = torch.softmax(corr_i.view(batch, ht * wd, h_d * w_d), dim=-1)

                crds_d = coords_grid(batch, h_d, w_d, device=corr_sm.device).view(batch, h_d * w_d, 2)
                flo = torch.einsum('b s m, b m f -> b s f', corr_sm, crds_d) * (ht / h_d) - crds

                flow_attn, conf, _ = self.cfp(self_corr=self_corr, corr_sm=corr_sm, thres=0.4 * 0.8 ** (ii + 1))

                flo = conf * flo + (1 - conf) * (flow_attn @ flo)
                flo = flo.view(batch, ht, wd, 2).permute(0, 3, 1, 2).contiguous()

                flo_s.append(flo)
            flos = torch.cat(flo_s, dim=1)

            flo_s = []
            for conv in self.conv_s:
                flo = conv(flos)
                flo_s.append(flo)
            flos = torch.cat(flo_s, dim=1)
            flo = self.conv_rd(flos)

            flo_0 = flo_0 + self.gamma * flo

        return flo_0, corr_fn
