from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from torch import Tensor

import torch
import torch.fft
import torch.nn as nn
import math
import numpy as np
from Utils.Fourier_Utils import gen_frequency_grid
from Utils.Light_Field_Utils import refocus_fft
from abc import ABC, abstractmethod

''' Create views from a light field ''' 
def _select_sais(lf: Tensor, u: Tensor, v: Tensor, u0: Tensor, v0: Tensor) -> Tensor:
    u0 = u0.flatten()
    v0 = v0.flatten()
    if u.flatten().equal(u0) and v.flatten().equal(v0):
        return lf

    num_views = u0.numel()
    ids = [torch.nonzero(torch.logical_and(u == u0[i], v == v0[i]), as_tuple=True) for i in range(num_views)]
    if not torch.all(torch.tensor([t[0].shape[0] == 1 for t in ids])):
        raise ValueError('The (u, v) coordinates of each views to be selected must be found '
                         'exactly once in the (u, v) coordinates of the input light field')

    lf = lf.permute([*range(2, lf.ndim - 2)] + [0, 1, -2, -1]).unsqueeze(-3)
    return torch.cat([lf[ids[i]][0] for i in range(num_views)], dim=-3)

'''
-----------------------------------------------
ABSTRACT CLASS FDLViewsParams
    Pytorch view parameters class for FDL model
-----------------------------------------------
'''
class FDLViewsParams(ABC, nn.Module):

    @abstractmethod
    def num_views(self):
        pass

    @abstractmethod
    def set_fdl_disparities(self, d: Tensor):
        pass

    @abstractmethod
    def make_fdl_data_matrix(self, w_x: Tensor, w_y: Tensor):
        pass

    def gen_views_from_light_field(self, lf, u, v):
        raise NotImplementedError()

'''
-------------------------------------------------
CLASS AllFocusParams
    All-in-focus views parameters of an FDL model
-------------------------------------------------
'''
class AllFocusParams(FDLViewsParams):
    def __init__(self, u: Tensor, v: Tensor):
        super(AllFocusParams, self).__init__()
        self.register_buffer('u', u.reshape(-1, 1).to(dtype=torch.float32))
        self.register_buffer('v', v.reshape(-1, 1).to(dtype=torch.float32))
        self.register_buffer('P_x', None)
        self.register_buffer('P_y', None)
    
    ''' Return the number of views '''  
    def num_views(self):
        return self.u.numel()
    
    ''' Prepare the FDL model with the fdl disparities '''  
    def set_fdl_disparities(self, d: Tensor):
        d = d.reshape(1, -1).to(device=self.u.device, dtype=torch.float32)
        self.P_x = self.u * d
        self.P_y = self.v * d
    
    ''' Create the FDL data-term matrix '''  
    def make_fdl_data_matrix(self, w_x: Tensor, w_y: Tensor):
        return torch.exp(2j * math.pi * (self.P_x * w_x + self.P_y * w_y))
    
    ''' Create views from a light field ''' 
    def gen_views_from_light_field(self, lf, u, v):
        return _select_sais(lf, u, v, self.u, self.v)


'''
---------------------------------------------------------------------------------------
CLASS DiscreteApertureFSParams
    Parameters of a Focal stack for an aperture defined as a dirac comb of an FDL model
---------------------------------------------------------------------------------------
'''
class DiscreteApertureFSParams(FDLViewsParams):
    def __init__(self, u: Tensor, v: Tensor, fs_disps: Tensor):
        super(DiscreteApertureFSParams, self).__init__()
        self.register_buffer('u', u.reshape(1, 1, -1).to(dtype=torch.float32))
        self.register_buffer('v', v.reshape(1, 1, -1).to(dtype=torch.float32))
        self.register_buffer('fs_disps', fs_disps.reshape(-1, 1, 1).to(dtype=torch.float32))
        self.register_buffer('P_x', None)
        self.register_buffer('P_y', None)

    ''' Return the number of views '''  
    def num_views(self):
        return self.fs_disps.numel()
    
    ''' Prepare the FDL model with the fdl disparities '''  
    def set_fdl_disparities(self, d: Tensor):
        d = d.reshape(1, -1, 1).to(device=self.u.device, dtype=torch.float32)
        self.P_x = self.u * (d - self.fs_disps)
        self.P_y = self.v * (d - self.fs_disps)

    ''' Create the FDL data-term matrix ''' 
    def make_fdl_data_matrix(self, w_x: Tensor, w_y: Tensor):
        A = torch.empty(w_x.shape[0:-2] + self.P_x.shape[:2], device=self.P_x.device, dtype=torch.complex64)
        for fs_id in range(self.P_x.shape[0]):
            A[..., fs_id, :] = torch.exp(2j * math.pi * (self.P_x[fs_id] * w_x + self.P_y[fs_id] * w_y)).mean(-1)
        return A
    
    ''' Create a focal stack from a light field ''' 
    def gen_views_from_light_field(self, lf, u, v):
        lf_select = _select_sais(lf, u, v, self.u, self.v)
        return refocus_fft(lf_select, self.u, self.v, self.fs_disps, 0)