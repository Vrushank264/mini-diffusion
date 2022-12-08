import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
import wandb

from model.Unet import UNet
from ddpm import DDPM


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def plot(ddpm_model, num_cls, ws, save_dir, epoch):

    ddpm_model.eval()
    num_samples = 4 * num_cls
    
    for w_i, w in enumerate(ws):

        pred, pred_arr = ddpm_model.sample(num_samples, (1,28,28), num_cls, w)
        real = torch.tensor(pred.shape).cuda()
        #print(pred.shape, real.shape)
        #combined = torch.cat([real, pred])
        grid = tv.utils.make_grid(pred, nrow = 10)

        grid_arr = grid.squeeze().detach().cpu().numpy()
        print(f'Grid Array Shape: {grid_arr.shape}')
        cv2.imwrite(f'{save_dir}/pred_epoch_{epoch}_w_{w}.png', grid_arr.transpose(1,2,0))
    
    ddpm_model.train()
    return grid_arr.transpose(1,2,0)
        
        
def train(unet, ddpm_model, loader, opt, criterion, scaler, num_cls, save_dir, ws, epoch, num_epochs):

    unet.train()
    ddpm_model.train()

    wandb.log({
        'Epoch': epoch
    })

    loop = tqdm(loader, position = 0, leave = True)
    loss_ = AverageMeter()

    for idx, (img, class_lbl) in enumerate(loop):

        img = img.cuda(non_blocking = True)
        lbl = class_lbl.cuda(non_blocking = True)
        
        opt.zero_grad(set_to_none = True)

        with torch.cuda.amp.autocast_mode.autocast():

            noise, x_t, ctx, timestep, ctx_mask = ddpm_model(img, lbl)
            pred = unet(x_t.half(), ctx, timestep.half(), ctx_mask.half())
            loss = criterion(noise, pred)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()


        loss_.update(loss.detach(), img.size(0))

        if idx % 200 == 0:
            wandb.log({
                'Loss': loss_.avg
            })
            print(loss_.avg)
        

    if epoch % 2 == 0:

      ddpm_model.eval()

      with torch.no_grad():
            n_sample = 4*num_cls
            for w_i, w in enumerate(ws):

                x1, xis = ddpm_model.sample(n_sample, (1, 28, 28), num_cls,w)


      fig, ax = plt.subplots(nrows = n_sample // num_cls, ncols = num_cls, sharex = True, sharey = True, figsize = (10, 4))

      def animate_plot(i, xis):

        print(f'GIF animation frame {i} of {xis.shape[0]}', end = '\r')
        plots = []

        for row in range(n_sample // num_cls):

          for col in range(num_cls):

            ax[row, col].clear()
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])

            plots.append(ax[row, col].imshow(-xis[i, (row*num_cls) + col, 0], cmap = 'gray', vmin = (-xis[i]).min(), vmax = (-xis[i]).max()))
        
        return plots

      ani = FuncAnimation(fig, animate_plot, fargs = [xis], interval = 200, blit = False, repeat = True, frames = xis.shape[0])
      ani.save(f'{save_dir}/epoch_{epoch}.gif', dpi = 100, writer = PillowWriter(fps = 5))
      print('Done')

      torch.save(ddpm_model.state_dict(), os.path.join(save_dir, f'ddpm.pth'))
      torch.save(unet.state_dict(), os.path.join(save_dir, f'unet.pth'))


def main():

    num_cls = 10
    num_epochs = 20
    save_dir = '/content/drive/MyDrive/Diffusion'
    unet = UNet(1, 128, num_cls).cuda()
    ddpm_model = DDPM(unet, (1e-4, 0.02)).cuda()

    #unet.load_state_dict(torch.load('/content/drive/MyDrive/Diffusion/unet.pth'))
    #ddpm_model.load_state_dict(torch.load('/content/drive/MyDrive/Diffusion/ddpm.pth'))

    tr = T.Compose([T.ToTensor()])
    dataset = tv.datasets.MNIST('/content/data', True, transform = tr, download = True)
    loader = DataLoader(dataset, batch_size = 64, shuffle = True, num_workers = 0)

    opt = torch.optim.Adam(list(ddpm_model.parameters()) + list(unet.parameters()), lr = 1e-4)
    criterion = nn.MSELoss()

    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    
    ws = [0.0, 0.5, 1.0]

    for epoch in range(num_epochs):

        train(unet, ddpm_model, loader, opt, criterion, scaler, num_cls, save_dir, ws, epoch, num_epochs)


if __name__ == '__main__':

    #wandb.init(project = 'MinDiffusion')
    main()
