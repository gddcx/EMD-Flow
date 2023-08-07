# -*- coding utf-8 -*-
import torch
import torch.nn as nn


def window_partition(fmap, window_size):
    """
    :param fmap: shape:B, H, W, C
    :param window_size: Wh, Ww
    :return: shape: B*nW, Wh*Ww, C
    """
    B, H, W, C = fmap.shape
    fmap = fmap.reshape(B, H//window_size[0], window_size[0], W//window_size[1], window_size[1], C)
    fmap = fmap.permute(0, 1, 3, 2, 4, 5).contiguous()
    fmap = fmap.reshape(B*(H//window_size[0])*(W//window_size[1]), window_size[0]*window_size[1], C)
    return fmap


def window_reverse(fmap, window_size, H, W):
    """
    :param fmap: shape:B*nW, Wh*Ww, dim
    :param window_size: Wh, Ww
    :param H: original image height
    :param W: original image width
    :return: shape: B, H, W, C
    """
    Bnw, _, dim = fmap.shape
    nW = (H // window_size[0]) * (W // window_size[1])
    fmap = fmap.reshape(Bnw//nW, H // window_size[0], W // window_size[1], window_size[0], window_size[1], dim)
    fmap = fmap.permute(0, 1, 3, 2, 4, 5).contiguous()
    fmap = fmap.reshape(Bnw//nW, H, W, dim)
    return fmap


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, scale=None):
        super().__init__()
        self.dim = dim
        self.scale = scale or dim ** (-0.5)
        self.q = nn.Linear(in_features=dim, out_features=dim)
        self.k = nn.Linear(in_features=dim, out_features=dim)
        self.v = nn.Linear(in_features=dim, out_features=dim)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, fmap, mask=None):
        """
        :param fmap1: B*nW, Wh*Ww, dim
        :param mask: nw, Wh*Ww, Ww*Wh
        :return: B*nW, Wh*Ww, dim
        """
        Bnw, WhWw, dim = fmap.shape
        q = self.q(fmap)
        k = self.k(fmap)
        v = self.v(fmap)

        q = q * self.scale
        attn = q @ k.transpose(1, 2)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.reshape(Bnw//nw, nw, WhWw, WhWw) + mask.unsqueeze(0)
            attn = attn.reshape(Bnw, WhWw, WhWw)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        x = attn @ v
        x = self.proj(x)
        return x


class GlobalAttention(nn.Module):
    def __init__(self, dim, scale=None):
        super().__init__()
        self.dim = dim
        self.scale = scale or dim ** (-0.5)
        self.q = nn.Linear(in_features=dim, out_features=dim)
        self.k = nn.Linear(in_features=dim, out_features=dim)
        self.v = nn.Linear(in_features=dim, out_features=dim)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, fmap1, fmap2, mask=None):
        """
        :param fmap1: B, H, W, C
        :param fmap2: B, H, W, C
        :param pe: B, H, W, C
        :return:
        """
        B, H, W, C = fmap1.shape
        q = self.q(fmap1)
        k = self.k(fmap2)
        v = self.v(fmap2)

        q, k, v = map(lambda x: x.reshape(B, H*W, C), [q, k, v])

        q = q * self.scale
        attn = q @ k.transpose(1, 2)
        if mask is not None:
            mask = mask.reshape(1, H * W, 1) | mask.reshape(1, 1, H * W)  # batch, hw, hw
            mask = mask.float() * -100.0
            attn = attn + mask
        attn = self.softmax(attn)
        x = attn @ v  # B, HW, C

        x = self.proj(x)

        x = x.reshape(B, H, W, C)
        return x


class SelfTransformerBlcok(nn.Module):
    def __init__(self, dim, mlp_scale, window_size, shift_size=None, norm=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        if norm == 'layer':
            self.layer_norm1 = nn.LayerNorm(dim)
            self.layer_norm2 = nn.LayerNorm(dim)
        else:
            self.layer_norm1 = nn.Identity()
            self.layer_norm2 = nn.Identity()

        self.self_attn = WindowAttention(dim=dim, window_size=window_size)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_scale),
            nn.GELU(),
            nn.Linear(dim * mlp_scale, dim)
        )

    def forward(self, fmap, mask=None):
        """
        :param fmap: shape: B, H, W, C
        :return: B, H, W, C
        """
        B, H, W, C = fmap.shape

        shortcut = fmap
        fmap = self.layer_norm1(fmap)

        if self.shift_size is not None:
            shifted_fmap = torch.roll(fmap, [-self.shift_size[0], -self.shift_size[1]], dims=(1, 2))
            if mask is not None:
                shifted_mask = torch.roll(mask, [-self.shift_size[0], -self.shift_size[1]], dims=(1, 2))
        else:
            shifted_fmap = fmap
            if mask is not None:
                shifted_mask = mask

        win_fmap = window_partition(shifted_fmap, window_size=self.window_size)
        if mask is not None:
            pad_mask = window_partition(shifted_mask.unsqueeze(-1), self.window_size)
            pad_mask = pad_mask.reshape(-1, self.window_size[0] * self.window_size[1], 1) \
                       | pad_mask.reshape(-1, 1, self.window_size[0] * self.window_size[1])

        if self.shift_size is not None:
            h_slice = [slice(0, -self.window_size[0]), slice(-self.window_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None)]
            w_slice = [slice(0, -self.window_size[1]), slice(-self.window_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None)]
            img_mask = torch.zeros([1, H, W, 1]).to(win_fmap.device)
            count = 0
            for h in h_slice:
                for w in w_slice:
                    img_mask[:, h, w, :] = count
                    count += 1
            win_mask = window_partition(img_mask, self.window_size)
            win_mask = win_mask.reshape(-1, self.window_size[0] * self.window_size[1])  # nW, Wh*Ww
            attn_mask = win_mask.unsqueeze(2) - win_mask.unsqueeze(1)  # nw, Wh*Ww, Wh*Ww
            if mask is not None:
                attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0).masked_fill((attn_mask != 0) | pad_mask, -100.0)
            else:
                attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0).masked_fill(attn_mask != 0, -100.0)
            attn_fmap = self.self_attn(win_fmap, attn_mask)
        else:
            if mask is not None:
                pad_mask = pad_mask.float()
                pad_mask = pad_mask.masked_fill(pad_mask != 0, -100.0).masked_fill(pad_mask == 0, 0.0)
                attn_fmap = self.self_attn(win_fmap, pad_mask)
            else:
                attn_fmap = self.self_attn(win_fmap, None)
        shifted_fmap = window_reverse(attn_fmap, self.window_size, H, W)

        if self.shift_size is not None:
            fmap = torch.roll(shifted_fmap, [self.shift_size[0], self.shift_size[1]], dims=(1, 2))
        else:
            fmap = shifted_fmap

        fmap = shortcut + fmap
        fmap = fmap + self.mlp(self.layer_norm2(fmap))  # B, H, W, C
        return fmap


class CrossTransformerBlcok(nn.Module):
    def __init__(self, dim, mlp_scale, norm=None):
        super().__init__()
        self.dim = dim

        if norm == 'layer':
            self.layer_norm1 = nn.LayerNorm(dim)
            self.layer_norm2 = nn.LayerNorm(dim)
            self.layer_norm3 = nn.LayerNorm(dim)
        else:
            self.layer_norm1 = nn.Identity()
            self.layer_norm2 = nn.Identity()
            self.layer_norm3 = nn.Identity()
        self.cross_attn = GlobalAttention(dim=dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_scale),
            nn.GELU(),
            nn.Linear(dim * mlp_scale, dim)
        )

    def forward(self, fmap1, fmap2, mask=None):
        """
        :param fmap1: shape: B, H, W, C
        :param fmap2: shape: B, H, W, C
        :return: B, H, W, C
        """
        shortcut = fmap1

        fmap1 = self.layer_norm1(fmap1)
        fmap2 = self.layer_norm2(fmap2)

        attn_fmap = self.cross_attn(fmap1, fmap2, mask)
        attn_fmap = shortcut + attn_fmap
        fmap = attn_fmap + self.mlp(self.layer_norm3(attn_fmap))  # B, H, W, C
        return fmap


class BasicLayer(nn.Module):
    def __init__(self, num_layer, dim, mlp_scale, window_size, cross=False):
        super().__init__()
        assert num_layer % 2 == 0, "The number of Transformer Block must be even!"
        self.blocks = nn.ModuleList()
        for n in range(num_layer):
            shift_size = None if n % 2 == 0 else [window_size[0]//2, window_size[1]//2]
            self.blocks.append(
                SelfTransformerBlcok(
                    dim=dim,
                    mlp_scale=mlp_scale,
                    window_size=window_size,
                    shift_size=shift_size,
                    norm='layer'
                )
            )
        # if cross:
        self.cross_transformer = CrossTransformerBlcok(dim=dim, mlp_scale=mlp_scale, norm='layer')
        self.cross = cross

    def forward(self, fmap1, fmap2, mask=None):
        """
        :param fmap1: B, H, W, C
        :param fmap2: B, H, W, C
        :return: B, H, W, C
        """
        B = fmap1.shape[0]
        fmap = torch.cat([fmap1, fmap2], dim=0)
        for blk in self.blocks:
            fmap = blk(fmap, mask)
        fmap1, fmap2 = torch.split(fmap, [B]*2, dim=0)
        if self.cross:
            fmap2 = self.cross_transformer(fmap2, fmap1, mask) + fmap2
            fmap1 = self.cross_transformer(fmap1, fmap2, mask) + fmap1
        return fmap1, fmap2