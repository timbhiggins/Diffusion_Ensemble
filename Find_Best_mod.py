import glob
import torch
import requests
import random
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch import Trainer
import os
from torch.utils.data import Dataset
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from sklearn.preprocessing import QuantileTransformer
from functools import lru_cache
from torch.utils.data import DataLoader
import math
import copy
import climpred
from pathlib import Path
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.amp import autocast
from torch.optim import Adam
from torchvision import transforms as T, utils
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from scipy.optimize import linear_sum_assignment
from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator
from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.version import __version__
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import re
from matplotlib.pyplot import figure
import utils_verif

""
gpu_id=0
""
def set_gpu(gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
if gpu_id >= 0:
    device = "cuda"
    set_gpu(gpu_id)
    print('device available :', torch.cuda.is_available())
    print('device count: ', torch.cuda.device_count())
    print('current device: ',torch.cuda.current_device())
    print('device name: ',torch.cuda.get_device_name())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')


lead_time = 72
run_name = f'Lead_'+str(lead_time)


config = {
        "input_channels": 1,
        "output_channels": 1,
        "context_image": True,
        "context_channels": 1,
        "num_blocks": [2, 2],
        "hidden_channels": 32,
        "hidden_context_channels": 8,
        "time_embedding_dim": 256,
        "image_size": 128,
        "noise_sampling_coeff": 0.85,
        "denoise_time": 970,
        "activation": "gelu",
        "norm": True,
        "subsample": 100000,
        "save_name": "model_weights.pt",
        "dim_mults": [4, 4],
        "base_dim": 32,
        "timesteps": 1000,
        "pading": "reflect",
        "scaling": "std",
        "optimization": {
            "epochs": 400,
            "lr": 0.01,
            "wd": 0.05,
            "batch_size": 32,
            "scheduler": True
        }
    }

model = Unet(
    channels = 1,
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True,
    self_condition = False
)


diffusion = GaussianDiffusion(
    model,
    image_size = (128, 128),
    timesteps = 1500,    # default is 1000 
    auto_normalize = True,
    objective = "pred_v",
    ddim_sampling_eta = 0.5,
)
diffusion.is_ddim_sampling = True

trainer = Trainer(
    diffusion,
    '/glade/derecho/scratch/timothyh/data/diffusion_forecasts/processed/lead_',
    int(lead_time),
    run_name,
    do_wandb = True,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 10000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False,           # whether to calculate fid during training
    max_grad_norm = 1.0,
)
def delle_2006(positions,numense):
    freqi = np.histogram(positions,bins=numense+1,density = True)[0]
    perfi = 1/(numense+1)
    score = np.sum(np.abs(freqi - perfi))
    print('######## RI: ', score, "#############")
    return freqi,perfi,score

models = '/glade/derecho/scratch/timothyh/diffusion/diff_results/model-544Lead_120.pt'
model_num = 1304
mean = 152.16729736328125
std = 147.735107421875
maxivt = 15.220598
minivt = -1.0300009
dataX = xr.open_dataset('/glade/derecho/scratch/timothyh/data/diffusion_forecasts/processed/lead_'+str(lead_time)+'/val.nc').forecast
datay = xr.open_dataset('/glade/derecho/scratch/timothyh/data/diffusion_forecasts/processed/lead_'+str(lead_time)+'/val.nc').analysis
dataX = (dataX-mean)/std
dataX = (dataX-minivt)/(maxivt-minivt)
datay = (datay-mean)/std
datay = (datay-minivt)/(maxivt-minivt)
X = torch.tensor(dataX.values).to(device).unsqueeze(1)
while model_num < 3000:
    dataX = xr.open_dataset('/glade/derecho/scratch/timothyh/data/diffusion_forecasts/processed/lead_'+str(lead_time)+'/val.nc').forecast
    datay = xr.open_dataset('/glade/derecho/scratch/timothyh/data/diffusion_forecasts/processed/lead_'+str(lead_time)+'/val.nc').analysis
    dataX = (dataX-mean)/std
    dataX = (dataX-minivt)/(maxivt-minivt)
    datay = (datay-mean)/std
    datay = (datay-minivt)/(maxivt-minivt)
    X = torch.tensor(dataX.values).to(device).unsqueeze(1)
    trainer.load(str(model_num))
    sampled_images = diffusion.sample(X[0:100], batch_size = 100, return_all_timesteps = False).detach().cpu().numpy()
    denoised = np.squeeze(sampled_images)

    all_denoised = torch.zeros([20,100,1,128,128]).to(device)

    all_denoised[0,:,:,:,:] = X[0:100,:]
    sampled_images = diffusion.sample(all_denoised[0,:,:,:,:], batch_size = 100, return_all_timesteps = False)
    all_denoised[1,:,:] = sampled_images
    ens_num = 1
    while ens_num < 20:
        sampled_images = diffusion.sample(all_denoised[0,:,:,:,:], batch_size = 100, return_all_timesteps = False)    
        all_denoised[ens_num,:,:,:,:] = sampled_images
        ens_num+=1
    all_denoised = all_denoised.detach().cpu().numpy()


    X = torch.tensor(dataX.values).to(device).unsqueeze(1)
    dataX = xr.open_dataset('/glade/derecho/scratch/timothyh/data/diffusion_forecasts/processed/lead_'+str(lead_time)+'/val.nc').forecast
    datay = xr.open_dataset('/glade/derecho/scratch/timothyh/data/diffusion_forecasts/processed/lead_'+str(lead_time)+'/val.nc').analysis
    print(np.max(all_denoised))
    denoised = all_denoised * (maxivt - minivt) + minivt
    denoised = (denoised*std + mean)


    rmse = np.sqrt((np.mean(denoised[:,:,0,:,:],axis=0)-xr.DataArray(datay[:100,:]))**2)
    print(np.max(denoised))
    ensemble_forecasts = denoised[:,:,0,:,:]
    Obs = datay[:100,:,:]
    ense = 21
    bins = ense # one less because we remove an ensemble... 
    fcast='F048'
    mod = 'Raw_gefs'
    Thresher=0

    Post_m = np.mean(ensemble_forecasts,axis=0)
    Post_s = np.std(ensemble_forecasts,axis=0)
    bounds = [[0,2300],[250,500],[500,10000]]


    rr2 = []
    for bb in bounds:
        print(np.max(ensemble_forecasts))
        Forecast_ense =  np.reshape(ensemble_forecasts,[20,100*128*128])
        m_All = np.mean(Forecast_ense,axis=0)
        Obs = np.reshape(np.array(datay[:100,:,:]),[100*128*128])
        print(np.max(Obs))
        print(np.max(Forecast_ense))
        rr2.append(utils_verif.ranker(Obs[(m_All>bb[0]) & (m_All<bb[1])],Forecast_ense[:,(m_All>bb[0]) & (m_All<bb[1])]))
    


    print('making figure... takes FOREVER')
    plt.figure(num=None, figsize=(20, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.hist(rr2,bins=bins,density = True,stacked=True,alpha=0.5,label=['0-250','250-500','500+'],color=sns.xkcd_palette(['grey','light blue','greyish blue']))
    plt.plot([-10,110],[1/(ense-1),1/(ense-1)],'k--')
    plt.legend(fontsize=20)
    plt.ylim([0,0.1])
    plt.xlim([0,bins+2])
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.title('mod_num='+str(model_num)+' rmse='+str(np.mean(rmse)))

    plt.savefig(str(model_num)+'.png',bbox_inches='tight')
    plt.show()

    print('perf ref:')
    fri, perfi, ss = delle_2006(np.concatenate([rr2[0],rr2[1],rr2[2]]),ense-1)
    print('perf ref:')
    
    model_num+=1
    
