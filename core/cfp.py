import torch
import torch.nn as nn


class CFP(nn.Module):
    def __init__(self, c_dim):
        super(CFP, self).__init__()
        self.self_corr = nn.Linear(c_dim, c_dim)

    def fetch_mask(self, self_corr, corr, thres=0.4):
        corr_mask = torch.max(corr, dim=-1)[0]
        confidence = torch.zeros_like(corr_mask)  
        confidence[corr_mask <= thres] = -100
        confidence = confidence.unsqueeze(1)  
        self_corr = self_corr + confidence  

        self_corr = torch.softmax(self_corr, dim=-1)
        corr_mask[corr_mask > thres] = 1.0
        return self_corr, corr_mask.unsqueeze(-1)

    def forward(self, inp=None, corr_sm=None, self_corr=None, thres=0.4):
        if self_corr is None:
            batch, ch, ht, wd = inp.shape
            inp = inp.reshape(batch, ch, ht * wd).permute(0, 2, 1).contiguous()
            inp = self.self_corr(inp)
            self_corr = (inp * (ch ** -0.5)) @ inp.transpose(1, 2)

        flow_attn, conf = self.fetch_mask(self_corr, corr_sm, thres=thres)

        return flow_attn, conf, self_corr
