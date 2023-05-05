import os
import torch

from Networks.UNet import UNetRes as net
from Networks.Utils import Utils_Model
from FDL.FDL_Model import FDLModel

'''
---------------------------------------------------------------------------------------------------
CLASS UnrolledADMMFDL
    A Pytorch module to unrolled the ADMM FDL optimization algorithm for light field reconstruction
---------------------------------------------------------------------------------------------------
'''
class UnrolledADMMFDL(torch.nn.Module):
    def __init__(self, nb_channels=3, nb_iteration=12, nb_fdl=30, rho_initial_val=1.0, device='cpu'):
        super(UnrolledADMMFDL,self).__init__()
        self.nb_iteration=nb_iteration
        self.device = device
        
        self.d_fdl = torch.arange(-2, 1, 3.0/nb_fdl)
        
        self.denoiser = net(in_nc=nb_channels*nb_fdl+1, out_nc=nb_channels*nb_fdl, nc=[128, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        self.fdl_model = FDLModel(self.d_fdl, device=device)

        self.rhos = torch.nn.ParameterList()
        for iteration in range(nb_iteration+1):
            self.rhos += [torch.nn.Parameter(data=torch.tensor(rho_initial_val), requires_grad=True)]  
    
    ''' Generate views from a set of FDL '''
    def generate_views(self, fdl, render_params):
        return self.fdl_model.fdl_render(fdl, render_params)

    ''' Execute the entire unrolled ADMM FDL optimization algorithm '''
    def forward(self, fdl, inputs, views_params):
        v = fdl
        u = 0
        y = torch.fft.fftn(inputs, dim=(-2, -1))
        for iteration in range(self.nb_iteration):
            x = self.fdl_model.fdl_proximal_operator(v - u / self.rhos[iteration], y, self.rhos[iteration], views_params)
            x_denoise = x + u / self.rhos[iteration]
            x_reshaped = x_denoise.view(*x_denoise.shape[0:1],-1,*x_denoise.shape[-2:])
            v = Utils_Model.test_mode(self.denoiser, x_reshaped, mode=2, refield=32, min_size=1024, modulo=16, iteration=iteration)
            v = v.view_as(x)
            u = u + self.rhos[iteration] * (x - v) 
        return v