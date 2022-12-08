import torch
import torch.nn as nn
import numpy as np
import math


def betas_for_alpha_bar(T, alpha_bar, max_beta):

    betas = []
    for i in range(T):
        t1 = i / T
        t2 = (i + 1) / T
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    
    betas = np.array(betas)
    return torch.from_numpy(betas)


def ddpm_schedule(beta_start, beta_end, T, scheduler_type = 'cosine'):

    assert beta_start < beta_end < 1.0
    
    if scheduler_type == 'linear':
        
        betas = torch.linspace(beta_start, beta_end, T)

    if scheduler_type == 'cosine':

        betas = betas_for_alpha_bar(T, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2, beta_end)

    if scheduler_type not in ['linear', 'cosine']:
        
        raise NotImplementedError(f'Unknown Beta Schedule {scheduler_type}.')
    
    sqrt_beta_t = torch.sqrt(betas)
    alpha_t = 1 - betas
    log_alpha_t = torch.log(alpha_t)
    sqrt_alpha_t_inv = 1 / torch.sqrt(alpha_t)

    alphabar_t = torch.cumsum(log_alpha_t, dim = 0).exp()
    sqrt_abar_t = torch.sqrt(alphabar_t)
    sqrt_abar_t1 = torch.sqrt(1 - alphabar_t)
    alpha_t_div_sqrt_abar = (1 - alpha_t) / sqrt_abar_t1

    return {
        'sqrt_beta_t': sqrt_beta_t,
        'alpha_t': alpha_t,
        'sqrt_alpha_t_inv': sqrt_alpha_t_inv,
        'alphabar_t': alphabar_t,
        'sqrt_abar_t': sqrt_abar_t,
        'sqrt_abar_t1': sqrt_abar_t1,
        'alpha_t_div_sqrt_abar': alpha_t_div_sqrt_abar
    } 



class DDPM(nn.Module):

    def __init__(self, model, betas, T = 500, dropout_p = 0.1):
        
        super().__init__() 
        self.model = model.cuda()

        for k, v in ddpm_schedule(betas[0], betas[1], T).items():
            self.register_buffer(k, v)

        self.T = T
        self.dropout_p = dropout_p

    
    def forward(self, x, cls):

        timestep = torch.randint(1, self.T, (x.shape[0], )).cuda()
        noise = torch.randn_like(x)

        x_t = (self.sqrt_abar_t[timestep, None, None, None] * x + self.sqrt_abar_t1[timestep, None, None, None] * noise)

        ctx_mask = torch.bernoulli(torch.zeros_like(cls) + self.dropout_p).cuda()
        
        return noise, x_t, cls, timestep / self.T, ctx_mask


    def sample(self, num_samples, size, num_cls, guide_w = 0.0):

        x_i = torch.randn(num_samples, *size).cuda() 
        c_i = torch.arange(0, num_cls).cuda()
        c_i = c_i.repeat(int(num_samples / c_i.shape[0]))

        ctx_mask = torch.zeros_like(c_i).cuda()
        c_i = c_i.repeat(2)
        ctx_mask = ctx_mask.repeat(2)
        ctx_mask[num_samples:] = 1.0

        #To Store intermediate results and create GIFs.
        x_is = []

        for i in range(self.T - 1, 0, -1):
            
            t_is = torch.tensor([i / self.T]).cuda()
            t_is = t_is.repeat(num_samples, 1, 1, 1)

            x_i = x_i.repeat(2, 1, 1, 1)   
            t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(num_samples, *size).cuda() if i > 1 else 0

            eps = self.model(x_i, c_i, t_is, ctx_mask)
            eps1 = eps[:num_samples]
            eps2 = eps[num_samples:]
            eps = (1 + guide_w)*eps1 - guide_w*eps2
            
            x_i = x_i[:num_samples]
            x_i = (self.sqrt_alpha_t_inv[i] * (x_i - eps*self.alpha_t_div_sqrt_abar[i]) + self.sqrt_beta_t[i] * z)

            if i % 25 == 0 or i == self.T - 1:
                x_is.append(x_i.detach().cpu().numpy())
            
        
        return x_i, np.array(x_is)

