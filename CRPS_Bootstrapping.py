import xarray as xr
import numpy as np
import climpred.metrics
import os
import gc
import dask.array as da



diff_path = '/glade/derecho/scratch/timothyh/data/diffusion_forecasts/completed/Denoised_IVT_day3.nc'
era_path = '/glade/derecho/scratch/timothyh/data/diffusion_forecasts/processed/lead_72/test.nc'
diff = xr.open_dataset(diff_path).Denoised_IVT
era = xr.open_dataset(era_path).analysis[:348,:,:]

era = era.assign_coords(lat = diff.lat, lon = diff.lon)
diff = diff.astype(np.float32)
era = era.astype(np.float32)
original_times = diff.time.values

Num_samples = 348
Nbs = 1000
P_Bootstrap = np.zeros([Nbs])
crps_a = climpred.metrics._crps(diff[:,:,:64,:64], era[:,:64,:64], dim=['member'])
crps_b = climpred.metrics._crps(diff[:,:,64:,64:], era[:,64:,64:], dim=['member'])
crps_c = climpred.metrics._crps(diff[:,:,64:,:64], era[:,64:,:64], dim=['member'])
crps_d = climpred.metrics._crps(diff[:,:,:64,64:], era[:,:64,64:], dim=['member'])
idxtotal = np.arange(348)
for i in range(Nbs):
    idx = np.random.choice(idxtotal,Num_samples,replace=True)
    crps_a_Boot = crps_a[idx,:,:]
    crps_b_Boot = crps_b[idx,:,:]
    crps_c_Boot = crps_c[idx,:,:]
    crps_d_Boot = crps_d[idx,:,:]

    P_Bootstrap[i] = (np.mean(crps_a_Boot)+np.mean(crps_b_Boot)+np.mean(crps_c_Boot)+np.mean(crps_d_Boot))/4
    print(f"Finished bootstrap iteration {i+1}/{Nbs}")

Booty = xr.DataArray(P_Bootstrap)
Booty.to_netcdf("/glade/derecho/scratch/timothyh/data/diffusion_forecasts/bootstrapping/DiffusionDay3.nc")