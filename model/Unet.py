import torch
import torch.nn as nn
import torch.nn.functional as fun
import math


class ResBlock(nn.Module):

    def __init__(self, in_c, out_c, is_res = False):
        
        super().__init__()
        self.is_res = is_res
        self.same_c = in_c == out_c
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size = 3, padding = 1),
                                   nn.BatchNorm2d(out_c),
                                   nn.GELU()
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(out_c, out_c, kernel_size = 3, padding = 1),
                                   nn.BatchNorm2d(out_c),
                                   nn.GELU()
                                   )
        
    def forward(self, x):
        
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        if self.is_res:
            if self.same_c:
                out = x + x2
            else:
                out = x1 + x2
            
            return out / math.sqrt(2)
        else:
            return x2
    

class UnetDown(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()

        self.block = nn.Sequential(ResBlock(in_c, out_c), 
                                   nn.Conv2d(out_c, out_c, kernel_size = 3, stride = 2, padding = 1),
                                   nn.BatchNorm2d(out_c),
                                   nn.GELU()
                                   )
    
    def forward(self, x):

        return self.block(x)
        

class UnetUp(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()

        self.block = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=1),
                                   ResBlock(out_c, out_c),
                                   ResBlock(out_c, out_c)
                                   )
    
    def forward(self, x, skip):

        x = torch.cat([x, skip], dim = 1)
        x = fun.interpolate(x, scale_factor=2)
        res = self.block(x)
        return res


class EmbeddingFC(nn.Module):

    def __init__(self, ip_dim, emb_dim):
        super().__init__()

        self.ip_dim = ip_dim
        self.block = nn.Sequential(nn.Linear(ip_dim, emb_dim),
                                   nn.GELU(),
                                   nn.Linear(emb_dim, emb_dim)
                                   )

        
    def forward(self, x):

        x = x.view(-1, self.ip_dim)
        res = self.block(x)
        return res


class UNet(nn.Module):

    def __init__(self, in_c, out_c, num_cls):
        super().__init__()
        
        self.in_c = in_c
        self.out_c = out_c
        self.num_cls = num_cls

        self.first_layer = ResBlock(in_c, out_c, is_res = True)
        self.down1 = UnetDown(out_c, out_c)
        self.down2 = UnetDown(out_c, out_c * 2)

        self.avg_pool = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.time_emb1 = EmbeddingFC(1, out_c * 2)
        self.time_emb2 = EmbeddingFC(1, out_c)
        self.ctx_emb1 = EmbeddingFC(num_cls, out_c * 2)
        self.ctx_emb2 = EmbeddingFC(num_cls, out_c)

        self.up = nn.Sequential(nn.ConvTranspose2d(out_c * 2, out_c * 2, 7, 7),
                                nn.GroupNorm(16, out_c * 2),
                                nn.ReLU()
                                )
        
        self.up1 = UnetUp(out_c *4, out_c)
        self.up2 = UnetUp(out_c * 2, out_c)

        self.penultimate_layer = nn.Sequential(nn.Conv2d(out_c * 2, out_c, kernel_size = 3, padding = 1),
                                               nn.GroupNorm(16, out_c),
                                               nn.ReLU()
                                               )
        
        self.last_layer = nn.Conv2d(out_c, self.in_c, kernel_size = 3, padding = 1)

    
    def forward(self, x, ctx, t, ctx_mask):

        '''
        x: Noisy Image
        ctx: Context label
        t: timestep
        ctx_mask: which samples to block context on

        '''

        x = self.first_layer(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        vec = self.avg_pool(d2)
        ctx = fun.one_hot(ctx, num_classes = self.num_cls).type(torch.float)

        ctx_mask = ctx_mask[:, None]
        ctx_mask = ctx_mask.repeat(1, self.num_cls)
        ctx_mask = (-1 * (1 - ctx_mask))
        ctx = ctx * ctx_mask

        c_emb1 = self.ctx_emb1(ctx).view(-1, self.out_c * 2, 1, 1)
        c_emb2 = self.ctx_emb2(ctx).view(-1, self.out_c, 1, 1)
        t_emb1 = self.time_emb1(t).view(-1, self.out_c * 2, 1, 1)
        t_emb2 = self.time_emb2(t).view(-1, self.out_c, 1, 1)

        up = self.up(vec)
        up1 = self.up1(up*c_emb1 + t_emb1, d2)
        up2 = self.up2(up1*c_emb2 + t_emb2, d1)
        res = self.penultimate_layer(torch.cat([up2, x], dim = 1))
        res = self.last_layer(res)

        return res
