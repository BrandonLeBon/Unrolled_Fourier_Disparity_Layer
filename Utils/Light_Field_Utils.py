from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Union, List, Optional, Sequence
if TYPE_CHECKING:
    from torch import Tensor

import numpy as np
import torch
import torch.nn.functional
import os
import math
import Utils.Image_Utils as img
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Utils.Fourier_Utils import hann_border, gen_frequency_grid
from Utils.Format_Utils import is_iterable, to_tensor

import warnings

import logging
from torchvision.utils import save_image

class _DisableLogger():
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, exit_type, exit_value, exit_traceback):
       logging.disable(logging.NOTSET)


# =======================================
# get image paths of files
# =======================================
def get_lf_paths(dataroot: str) -> Optional[List[str]]:
    paths = None  # return None if dataroot is None
    if dataroot is not None:
        paths = sorted(_get_paths_from_lfs(dataroot))
    return paths


def _get_paths_from_lfs(path: str) -> List[str]:
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    lfs = []
    for name in sorted(os.listdir(path)):
        lf_path = os.path.join(path, name)
        if os.path.isdir(lf_path):
            lfs.append(lf_path)
    assert lfs, '{:s} has no valid folder'.format(path)
    return lfs



def load_lf(
        path: str,
        LF_name_prefix: str,
        u_range: Sequence[int],
        v_range: Sequence[int],
        ext: str = None,
        n_channels: int = 3
) -> Tensor:
    """
    Load light field data as a 6D tensor with dimensions:
        0. Batch (singleton).
        1. Channels.
        2. Angular vertical dimension (v).
        3. Angular horizontal dimension (u).
        4. Spatial vertical dimension (y).
        5. Spatial horizontal dimension (x).
    :param path: path to the light field data folder. The folder must contain image files of each view named as
    "[Path]/[LF_name_prefix][v]_[u].[ext]", for a view (u,v).
    :param LF_name_prefix: prefix of the image filenames before the view indices digits.
    :param u_range: list of indices of u (horizontal angular dimension).
    :param v_range: list of indices of v (horizontal angular dimension).
    :param ext: file extension without the dot (optional).
    If not specified or None, any image file exension is accepted.
    :param n_channels: number of color channels to read (1 or 3).
    For color images using n_channels=1 will convert to gray.
    For grayscale images, using n_channels=3 will output 3 identical channels.
    """
    file_list = os.listdir(path)
    ext_reg = r"("+'|'.join([ext[1:] for ext in img.IMG_EXTENSIONS])+r")" if ext is None else ext
    for u_idx, u in enumerate(u_range):
        for v_idx, v in enumerate(v_range):
            prog = re.compile(r"^" + LF_name_prefix + r"0*" + str(v) + r"_0*" + str(u) + r"\." + ext_reg + r"$")
            f_name = next(filter(prog.search, file_list), None)
            if f_name is None:
                raise RuntimeError(
                    f'Can\'t find image file of light field view (u={str(u)},v={str(v)}) in folder {path}')
            tmp = img.imread(os.path.join(path, f_name), n_channels)
            if u_idx == 0 and v_idx == 0:
                sz = tmp.shape
                lf = torch.empty(sz[0], sz[1], len(v_range), len(u_range), sz[2], sz[3], dtype=tmp.dtype)
            lf[:, :, v_idx, u_idx, :, :] = tmp
    return lf


def save_lf(
        lf: Tensor,
        path,
        u_range: Sequence[int] = None,
        v_range: Sequence[int] = None,
        LF_name_prefix='',
        ext: str = 'png'
):
    """
    Save light field data:
    :param lf: light field 6D tensor with dimensions:
        0. Batch (singleton).
        1. Channels.
        2. Angular vertical dimension (v).
        3. Angular horizontal dimension (u).
        4. Spatial vertical dimension (y).
        5. Spatial horizontal dimension (x).
    :param path: path to the light field data folder to save.
    :param u_range: list of indices of u (horizontal angular dimension) only used for generating indices in filenames.
    :param v_range: list of indices of v (horizontal angular dimension) only used for generating indices in filenames.
    :param LF_name_prefix: prefix of the image filenames before the view indices digits.
    :param ext: file extension without the dot (optional). Default: 'png'.
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        warnings.warn('The light field folder already exists. Image data may be overwritten.')
    if u_range is None:
        u_range = range(lf.shape[-3])
    if v_range is None:
        v_range = range(lf.shape[-4])
    for u_idx, u in enumerate(u_range):
        for v_idx, v in enumerate(v_range):
            img.imsave(lf[:, :, v_idx, u_idx, :, :], os.path.join(path, f'{LF_name_prefix}{v:03d}_{u:03d}.{ext}'))


def save_seq(
        seq: Tensor,
        path,
        frame_range: Sequence[int] = None,
        seq_name_prefix='',
        ext: str = 'png'
):
    """
    Save image sequence data.
    :param seq: image sequence 5D tensor with dimensions:
        0. Batch (singleton).
        1. Channels.
        2. Frames dimension.
        3. Spatial vertical dimension (y).
        4. Spatial horizontal dimension (x).
    :param path: path to the sequence data folder. The folder must contain image files of each view named as
    "[Path]/[seq_name_prefix]_[f].[ext]", for a frame f.
    :param frame_range: list of frames indices only used for generating indices in filenames.
    :param seq_name_prefix: prefix of the image filenames before the view indices digits.
    :param ext: file extension without the dot (optional). Default: 'png'.
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        warnings.warn('The sequence folder already exists. Image data may be overwritten.')
    if frame_range is None:
        frame_range = range(seq.shape[-3])
    for f_idx, f in enumerate(frame_range):
        img.imsave(seq[:, :, f_idx, :, :], os.path.join(path, f'{seq_name_prefix}{f:03d}.{ext}'))


def pad_lf(lf: Tensor, pad: Union[int, Sequence[int]], mode: str = 'reflect', use_window: bool = True):
    """
    Pads the Light Field tensor in spatial dimensions with number of pixels defined in 'pad'.
    :param lf: Light Field tensor with dimensions (batch x channels x V x U x Y x X).
    :param pad: int or tuple of ints, number of pixels to pad on each side: (left, right, top, bottom).
    :param mode: padding mode (from pytorch padding function)
    :param use_window: set to True to apply Hann windowing to the padded borders.
    """
    if not is_iterable(pad):
        pad = (pad, pad, pad, pad)
    lf_shape = lf.shape[:-2] + (lf.shape[-2] + pad[2] + pad[3], lf.shape[-1] + pad[0] + pad[1])
    lf = torch.nn.functional.pad(lf.view(-1, lf.shape[-3], lf.shape[-2], lf.shape[-1]), pad, mode=mode).view(lf_shape)
    if use_window:
        lf *= hann_border(lf.shape[-1], lf.shape[-2], pad, device=lf.device)
    return lf


def refocus_fft(
        lf: Tensor,
        u: Sequence[Union[float, int]],
        v: Sequence[Union[float, int]],
        d: Sequence[Union[float, int]],
        pad: Union[int, Sequence[int]],
        n_views_batch: int = 32,
        n_refocus_batch: int = 10,
) -> Tensor:
    n_views = int(np.prod(lf.shape[2:-2]))
    n_d = len(d)
    u = to_tensor(u).reshape(-1, 1, 1)
    v = to_tensor(v).reshape(-1, 1, 1)
    if u.shape[0] != n_views or v.shape[0] != n_views:
        raise ValueError(
            f'The number of elements in u (={u.shape[0]}) and v (={v.shape[0]}) must be equal to the '
            f'number of views of the input light field (={n_views}).')
    if not is_iterable(pad):
        pad = (pad, pad, pad, pad)
    lf = pad_lf(lf, pad)

    spatial_shape = lf.shape[-2:]
    lf = torch.fft.rfftn(lf, dim=(-2, -1))
    spatial_shape_half = lf.shape[-2:]
    w_x, w_y = gen_frequency_grid(spatial_shape[1], spatial_shape[0], half_x_dim=True, device=lf.device)
    # Prepare for processing dimensions in the order: ... n_d x n_views x spatial_vertical x spatial_horizontal
    lf = lf.view(lf.shape[:2] + (1, n_views) + spatial_shape_half)
    w_x = w_x.unsqueeze(0)
    w_y = w_y.unsqueeze(0)
    d = to_tensor(d, device=lf.device).view(-1, 1, 1, 1)

    # r: ... x n_d x spatial_vertical x spatial_horizontal
    r = torch.zeros(lf.shape[:2] + (n_d,) + spatial_shape_half, dtype=torch.complex64, device=lf.device)
    for d_st in range(0, n_d, n_refocus_batch):
        d_ids = slice(d_st, min(n_d, d_st + n_refocus_batch))
        for view_st in range(0, n_views, n_views_batch):
            view_ids = slice(view_st, min(n_views, view_st + n_views_batch))
            shift = torch.exp(-2j * math.pi * d[d_ids, ...] * (w_x * u[view_ids] + w_y * v[view_ids]))

            r[..., d_ids, :, :] += (lf[..., view_ids, :, :] * shift).sum(-3)
    r /= n_views

    r = torch.fft.irfftn(r, dim=(-2, -1), s=spatial_shape)
    return r[..., pad[2]: r.shape[-2]-pad[3], pad[0]: r.shape[-1]-pad[1]]
    

def refocus_fft_matrix(
        lf: Tensor,
        u: Sequence[Union[float, int]],
        v: Sequence[Union[float, int]],
        d: Sequence[Union[float, int]],
        pad: Union[int, Sequence[int]],
        n_views_batch: int = 32,
        n_refocus_batch: int = 10,
) -> Tensor:
    n_views = int(np.prod(lf.shape[2:-2]))
    n_d = len(d)
    u = to_tensor(u).reshape(-1, 1, 1)
    v = to_tensor(v).reshape(-1, 1, 1)
    if u.shape[0] != n_views or v.shape[0] != n_views:
        raise ValueError(
            f'The number of elements in u (={u.shape[0]}) and v (={v.shape[0]}) must be equal to the '
            f'number of views of the input light field (={n_views}).')
    if not is_iterable(pad):
        pad = (pad, pad, pad, pad)
    lf = pad_lf(lf, pad)

    spatial_shape = lf.shape[-2:]
    lf = torch.fft.rfftn(lf, dim=(-2, -1))
    spatial_shape_half = lf.shape[-2:]
    w_x, w_y = gen_frequency_grid(spatial_shape[1], spatial_shape[0], half_x_dim=True, device=lf.device)

    # Prepare for processing dimensions in the order: ... n_d x n_views x spatial_vertical x spatial_horizontal
    #lf = lf.view(lf.shape[:2] + (1,n_views) + spatial_shape_half).contiguous()
    w_x = w_x.unsqueeze(0)
    w_y = w_y.unsqueeze(0)
    d = to_tensor(d, device=lf.device).view(-1, 1, 1, 1)

    #lf = lf.view(*lf.shape[:-2],-1).permute(0,1,4,3,2)
    shift = torch.exp(-2j * math.pi * d * (w_x * u + w_y * v)) / n_views
    shift = shift.view(*shift.shape[:-2],-1).permute(2,0,1)
    #r = r.view(*r.shape[:-1],*spatial_shape_half)
    """
    for d_st in range(0, n_d, n_refocus_batch):
        d_ids = slice(d_st, min(n_d, d_st + n_refocus_batch))
        for view_st in range(0, n_views, n_views_batch):
            view_ids = slice(view_st, min(n_views, view_st + n_views_batch))
            shift = torch.exp(-2j * math.pi * d[d_ids, ...] * (w_x * u[view_ids] + w_y * v[view_ids]))
            print(shift.size())
            print(lf.size())
            import sys
            sys.exit()
            r[..., d_ids, :, :] += (lf[..., view_ids, :, :] * shift).sum(-3)
    """
    #r /= n_views

    #r = torch.fft.irfftn(r, dim=(-2, -1), s=spatial_shape)
    #return r[..., pad[2]: r.shape[-2]-pad[3], pad[0]: r.shape[-1]-pad[1]]
    return shift



def anim_lf(lf: Tensor, scan_mode: str = 'z_horz', frame_rate: int = 30):
    scan_modes = ['z_horz', 'z_vert', 's_horz', 's_vert']

    nU = lf.shape[3]
    nV = lf.shape[2]
    fig = plt.figure()

    def gen_function():
        while True:
            if scan_mode == 'z_horz':
                for v in range(nV):
                    for u in range(nU):
                        yield u, v
            elif scan_mode == 'z_vert':
                for u in range(nU):
                    for v in range(nV):
                        yield u, v
            elif scan_mode == 's_horz':
                backward = 0
                for v in range(nV):
                    for u in range(nU)[::1-(backward << 1)]:
                        yield u, v
                    backward ^= 1
            elif scan_mode == 's_vert':
                backward = 0
                for u in range(nU):
                    for v in range(nV)[::1-(backward << 1)]:
                        yield u, v
                    backward ^= 1
            else:
                raise ValueError(f'Invalid scan mode: accepted values are: {scan_modes}')

    def update(uv_id):
        with _DisableLogger():  # remove imshow warning
            im.set_data(lf[0, :, uv_id[1], uv_id[0], :, :].permute(1, 2, 0))
        return im,

    u0, v0 = next(gen_function())
    im = plt.imshow(lf[0, :, v0, u0, :, :].permute(1, 2, 0))

    ani = animation.FuncAnimation(fig, update, frames=gen_function, interval=1000//frame_rate, blit=True)

    plt.show(block=False)
    return ani


def anim_seq(seq: Tensor, scan_mode: str = 'loop', frame_rate: int = 30):
    scan_modes = ['loop', 'rewind']

    n_frames = seq.shape[2]
    fig = plt.figure()

    def gen_function():
        while True:
            if scan_mode == 'loop':
                for f in range(n_frames):
                    yield f
            elif scan_mode == 'rewind':
                backward = 0
                for f in range(n_frames)[::1-(backward << 1)]:
                    yield f
                backward ^= 1
            else:
                raise ValueError(f'Invalid scan mode: accepted values are: {scan_modes}')

    def update(f_id):
        with _DisableLogger():  # remove imshow warning
            im.set_data(seq[0, :, f_id, :, :].permute(1, 2, 0))
        return im,

    f0 = next(gen_function())
    im = plt.imshow(seq[0, :, f0, :, :].permute(1, 2, 0))

    ani = animation.FuncAnimation(fig, update, frames=gen_function, interval=1000//frame_rate, blit=True)

    plt.show(block=False)
    return ani



# ---------------------------------------------
# Metrics for np3 image format
# ---------------------------------------------
# ----------
# PSNR
# ----------
def calculate_psnr_lf(lf1, lf2, border=0):
    psnr = 0
    n_views = np.prod(lf1.shape[2:-2])
    lf1 = lf1.reshape(lf1.shape[0], lf1.shape[1], n_views, lf1.shape[-2], lf1.shape[-1])
    lf2 = lf2.reshape(lf1.shape[0], lf1.shape[1], n_views, lf1.shape[-2], lf1.shape[-1])
    for view in range(n_views):
        lf1_view = img.f32_to_uint8(img.tensor4_to_np3(lf1[:, :, view, :, :]))
        lf2_view = img.f32_to_uint8(img.tensor4_to_np3(lf2[:, :, view, :, :]))
        psnr += img.calculate_psnr(lf1_view, lf2_view, border)
    psnr /= n_views
    return psnr

# ----------
# SSIM
# ----------
def calculate_ssim_lf(lf1, lf2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    psnr = 0
    n_views = np.prod(lf1.shape[2:-2])
    lf1 = lf1.reshape(lf1.shape[0], lf1.shape[1], n_views, lf1.shape[-2], lf1.shape[-1])
    lf2 = lf2.reshape(lf1.shape[0], lf1.shape[1], n_views, lf1.shape[-2], lf1.shape[-1])
    for view in range(n_views):
        lf1_view = img.f32_to_uint8(img.tensor4_to_np3(lf1[:, :, view, :, :]))
        lf2_view = img.f32_to_uint8(img.tensor4_to_np3(lf2[:, :, view, :, :]))
        psnr += img.calculate_ssim(lf1_view, lf2_view, border)
    psnr /= n_views
    return psnr


# ---------------------------------------------
# Processing tools for Tensor4 image format
# ---------------------------------------------


########################################################################################################################
def main():
    import matplotlib.pyplot as plt
    import torch.fft
    #import utils.light_field_utils as lfu
    #lf_path = 'C:/Lightfield_project/FDL-Toolbox/Demo/Illum_Field'
    #lf_prefix = ''
    #u_range = range(3, 12)
    #v_range = range(3, 12)
    lf_path = '/nfs/nas4/sirocco_clim/sirocco_clim_image/data-guillaume/synthetic-dataset/training/Pens'
    lf_prefix = 'lf_'  # ''
    u_range = range(1, 10)  # range(3, 12)
    v_range = range(1, 10)  # range(3, 12)
    LF = load_lf(lf_path, lf_prefix, u_range, v_range)
    LF = LF[..., 1:-1, 1:-1]
    v, u = torch.meshgrid(torch.arange(4, -5, -1), torch.arange(-4, 5))
    d = torch.arange(-1, 1, .1)
    pad = (15, 15, 15, 15)
    r = refocus_fft(LF, u, v, d, 0)
    plt.figure()
    plt.imshow(r[0, :, 0].permute(1, 2, 0))
    plt.figure()
    plt.imshow(r[0, :, -1].permute(1, 2, 0))
    plt.figure()
    plt.imshow(LF[0, :, 4, 4].permute(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    main()
