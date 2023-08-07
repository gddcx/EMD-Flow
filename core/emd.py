# -*- coding utf-8 -*-
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import coords_grid
from update import BasicUpdateBlock
from corr import CorrBlock
from encoders import twins_svt_large, twins_svt_small_context
from convex_upsample import UpSampleMask8, UpSampleMask4

from mma import MMA


try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass



class EMD(nn.Module):
    def __init__(self, args):
        super().__init__()
        c_dim = 128
        self.args = args
        if 'dropout' not in self.args:
            self.args.dropout = 0
        self.fnet = twins_svt_large(pretrained=True)
        self.cnet = twins_svt_small_context(pretrained=True)

        self.mma = MMA(c_dim)

        self.up_mask8 = UpSampleMask8(c_dim)
        self.up_mask4 = UpSampleMask4(c_dim)

        args.corr_levels = 1
        args.corr_radius = 4
        self.inp_conv1x1 = nn.Conv2d(in_channels=c_dim, out_channels=c_dim, kernel_size=1)
        self.net_conv1x1 = nn.Conv2d(in_channels=c_dim, out_channels=c_dim, kernel_size=1)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=c_dim, input_dim=c_dim)

        print(f' -- Train iterations: 1/8: {self.args.iters8}, 1/4: {self.args.iters4}')

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def up_sample_flow8(self, flow, mask):
        B, _, H, W = flow.shape
        flow = torch.nn.functional.unfold(8 * flow, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1])
        flow = flow.reshape(B, 2, 9, 1, 1, H, W)  # batch, 2, 9, 1, 1, H, W
        mask = mask.reshape(B, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = torch.sum(flow * mask, dim=2)   # batch, 2, 8, 8, H, W
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3).contiguous()
        up_flow = up_flow.reshape(B, 2, H * 8, W * 8)
        return up_flow

    def up_sample_flow4(self, flow, mask):
        B, _, H, W = flow.shape
        flow = torch.nn.functional.unfold(4 * flow, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1])
        flow = flow.reshape(B, 2, 9, 1, 1, H, W)  # batch, 2, 9, 1, 1, H, W
        mask = mask.reshape(B, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = torch.sum(flow * mask, dim=2)   # batch, 2, 8, 8, H, W
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3).contiguous()
        up_flow = up_flow.reshape(B, 2, H * 4, W * 4)
        return up_flow

    def initialize_flow8(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)
        return coords0, coords1

    def initialize_flow4(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//4, W//4, device=img.device)
        coords1 = coords_grid(N, H//4, W//4, device=img.device)
        return coords0, coords1

    def forward(self, image1, image2, test_mode=False, refinemet=True):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        with autocast(enabled=self.args.mixed_precision):
            fmap8, fmap4 = self.fnet(torch.cat([image1, image2], dim=0))
            c_out8, inp4 = self.cnet(image1)
            inp8 = self.inp_conv1x1(c_out8)
            net = self.net_conv1x1(c_out8)

        fmap1_8, fmap2_8 = torch.chunk(fmap8, chunks=2, dim=0)
        fmap1_4, fmap2_4 = torch.chunk(fmap4, chunks=2, dim=0)

        flo_0, corr_fn = self.mma(fmap1_8, fmap2_8)

        flow_list = []
        if not test_mode:
            up_flow_mask = self.up_mask8(net)
            flow_up = self.up_sample_flow8(flo_0, up_flow_mask)
            flow_list.append(flow_up)

        coords0, coords1 = self.initialize_flow8(image1)  # batch, h, w, 2
        coords0 = coords0.permute(0, 3, 1, 2).contiguous()
        coords1 = coords1.permute(0, 3, 1, 2).contiguous()
        coords1 = coords1 + flo_0

        for itr in range(self.args.iters8):
            first_step = False if itr != 0 else True
            coords1 = coords1.detach()
            corr = corr_fn(coords1)

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, delta_flow = self.update_block(net, inp8, corr, flow, first_step=first_step)
            up_mask = self.up_mask8(net)

            coords1 = coords1 + delta_flow
            flow_up = self.up_sample_flow8(coords1 - coords0, up_mask)
            flow_list.append(flow_up)

        if refinemet:
            flow4 = torch.nn.functional.interpolate(2 * (coords1 - coords0), scale_factor=2, mode='bilinear', align_corners=False)
            coords0, coords1 = self.initialize_flow4(image1)  # batch, h, w, 2
            coords0 = coords0.permute(0, 3, 1, 2).contiguous()  # batch, 2, h, w
            coords1 = coords1.permute(0, 3, 1, 2).contiguous()
            coords1 = coords1 + flow4

            net = torch.nn.functional.interpolate(net, scale_factor=2, mode='bilinear', align_corners=False)
            corr_fn4 = CorrBlock(fmap1_4, fmap2_4, num_levels=self.args.corr_levels, radius=self.args.corr_radius)

            for itr in range(self.args.iters4):
                coords1 = coords1.detach()
                corr = corr_fn4(coords1)

                flow = coords1 - coords0
                with autocast(enabled=self.args.mixed_precision):
                    net, delta_flow = self.update_block(net, inp4, corr, flow, first_step=False)
                up_mask = self.up_mask4(net)

                coords1 = coords1 + delta_flow
                flow_up = self.up_sample_flow4(coords1 - coords0, up_mask)
                flow_list.append(flow_up)

        if test_mode:
            return flow_list[-1]
        return flow_list


