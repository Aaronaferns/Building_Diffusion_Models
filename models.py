import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from einops import einsum

'''
Def a Unet model
    Def Downsample
    Def Upsample
    
    
    Things to consider: 
        BatchNorm: batch statistics, bad for small batchsizes
        Use: GroupNorm
        
        Activation: ReLU:
        use: SiLU/Switch
        
        Blocks: use Resnet instead of plain conv
        
        Attention: Selectively for (low-res)
        
        Conditioning: TimeStep embeddings
'''

def nonlinearity():
    return nn.SiLU()
    
def group_norm(C, max_groups=32, eps=1e-5):
    for g in (32, 16, 8, 4, 2, 1):
        if g <= max_groups and C % g == 0:
            return nn.GroupNorm(g, C, eps=eps, affine=True)
    return nn.GroupNorm(1, C, eps=eps, affine=True) 

#ResBlock
'''
           residual branch
x ──┬──► GN → SiLU → Conv → (+temb) → GN → SiLU → Dropout → Conv ──┐
    │                                                              │
    └──► (identity OR 1×1/3×3 projection if needed) ───────────────┘
                               add → output
'''
class ResBlock(nn.Module):
    def __init__(self,  *, in_ch, out_ch, temb_dim, dropout):
        super().__init__()
        self.out_ch = out_ch
        self.norm1 = group_norm(in_ch)
        self.norm2 = group_norm(out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.temb_proj = nn.Linear(temb_dim, out_ch) 
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nonlinearity()
        
        # conv2 is zero initialized so the entire resnet block starts as identity out~x
        nn.init.zeros_(self.conv2.weight) 
        if self.conv2.bias is not None: nn.init.zeros_(self.conv2.bias)
          
    def forward(self, x, temb):
        B, C, H, W  = x.shape 
        h = x  # for skip connection
        h = self.conv1(self.act(self.norm1(h))) # x -> GN -> SiLU -> Conv
        
        # temb.shape: B, temb_dim 
        temb = self.temb_proj(self.act(temb))
        temb = temb.reshape(B, self.out_ch, 1, 1)
        
        
        h += temb
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return self.skip(x) + h



#DownBlock

class DownBlock(nn.Module):
    def __init__(self, *, chs, with_conv=True):
        super().__init__()
        if with_conv: self.layer = nn.Conv2d(chs, chs, kernel_size=3, stride = 2, padding = 1)
        else:         self.layer = nn.AvgPool2d(kernel_size=2, stride = 2)
        
    def forward(self, x):
        return self.layer(x)

#UpBlock

class UpBlock(nn.Module):
    def __init__(self, *, chs, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(chs, chs, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, C, H, W)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x
    
#AttnBlock
class AttnBlock(nn.Module): 
    def __init__(self, *, chs):
        super().__init__()
        self.norm = group_norm(chs)
        
        self.q = nn.Conv2d(chs, chs, kernel_size=1)
        self.k = nn.Conv2d(chs, chs, kernel_size=1)
        self.v = nn.Conv2d(chs, chs, kernel_size=1)
        self.out = nn.Conv2d(chs, chs, kernel_size=1)
        
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None: 
            nn.init.zeros_(self.out.bias)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)  # B, C, H, W
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        
        # we need to [B, H*W, C] for q and v
        q = q.permute(0,2,3,1).reshape(B, H*W, C)
        k = k.permute(0,2,3,1).reshape(B, H*W, C)
        v = v.permute(0,2,3,1).reshape(B, H*W, C)
        
        
        pre_softmax = einsum(q, k, "b n c, b m c -> b n m")
        pre_softmax = pre_softmax*(C**-0.5)
        sims = F.softmax(pre_softmax, dim=-1)
        attn_scores = einsum(sims, v, "b n m, b m c -> b n c")
        h_attn = attn_scores.reshape(B, H, W, C).permute(0,3,1,2)
        h_attn = self.out(h_attn)
        return x + h_attn
        
        
              
# model
    
class Unet(nn.Module):
    def __init__(self, *, in_resolution, input_ch, ch, output_ch, num_res_blocks, temb_dim, attn_res, dropout = 0., ch_mult=[1,2,4,8]):
        super().__init__()
        self.act = nonlinearity()
        self.conv_in = nn.Conv2d( in_channels = input_ch, out_channels = ch, kernel_size = 3, stride = 1, padding = 1)
        self.temb_dim = temb_dim
        
        # Down path
        self.contracting_path = nn.ModuleList()
        
        curr_res = in_resolution
        in_ch = ch
        skip_ch = []
        for i in range(len(ch_mult)):
            out_ch = ch*ch_mult[i]
            for j in range(num_res_blocks):
                self.contracting_path.append(ResBlock( in_ch=in_ch, out_ch=out_ch, temb_dim=temb_dim, dropout=dropout))
                skip_ch.append(out_ch)
                #update in channel to out ch after first resnet block
                in_ch = out_ch  
                # if current resolution is in attn_res array, add an attention block
                if curr_res in attn_res:
                    self.contracting_path.append(AttnBlock(chs = out_ch))
            # downsample everytime except at the end
            if i != len(ch_mult) - 1:
                self.contracting_path.append(DownBlock(chs = out_ch, with_conv=True))
                curr_res//=2
        
        #Middle 
        self.middle = nn.ModuleList()
        self.middle.append(ResBlock( in_ch=out_ch, out_ch=out_ch, temb_dim=temb_dim, dropout=dropout))
        self.middle.append(AttnBlock(chs = out_ch))
        self.middle.append(ResBlock( in_ch=out_ch, out_ch=out_ch, temb_dim=temb_dim, dropout=dropout))
        
        curr_res = in_resolution // (2 ** (len(ch_mult)-1))

        #Up path
        self.expanding_path = nn.ModuleList()
        
        for i in range(len(ch_mult)-1, -1, -1): 
            in_ch = out_ch  
            out_ch = ch_mult[i]*ch 
            for j in range(num_res_blocks):
                self.expanding_path.append(ResBlock( in_ch=in_ch+skip_ch.pop(), out_ch=out_ch, temb_dim=temb_dim, dropout=dropout))
                in_ch = out_ch
                if curr_res in attn_res:
                    self.expanding_path.append(AttnBlock(chs = out_ch))
            # upsample everytime except at the end
            if i != 0:
                self.expanding_path.append(UpBlock(chs = out_ch, with_conv=True))
                curr_res*=2
        
        self.conv_out = nn.Conv2d( in_channels = out_ch, out_channels = output_ch, kernel_size = 3, stride = 1, padding = 1)
        self.norm_out = group_norm(out_ch)
        
        
        self.temb_l1 = nn.Linear(temb_dim, temb_dim*4)
        self.temb_l2 = nn.Linear(temb_dim*4, temb_dim)
        
        
        
    
    def forward(self, x, t):
        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.temb_l2(self.act(self.temb_l1(temb)))
        
        
        x = self.conv_in(x)
        
        skips = []
        h = x
        for layer in self.contracting_path:
            
            if isinstance(layer,ResBlock):
                h = layer(h,temb) 
                skips.append(h)    
            else:
                h = layer(h)  
            if isinstance(layer,AttnBlock):
                skips[-1] = h
            
        
        for layer in self.middle:
            h = layer(h,temb) if isinstance(layer,ResBlock) else layer(h) 
        
        for layer in self.expanding_path:
            if isinstance(layer,ResBlock):
                h = torch.cat([h , skips.pop()],dim = 1)
            h = layer(h,temb) if isinstance(layer,ResBlock) else layer(h) 
        
        return self.conv_out(self.act(self.norm_out(h)))