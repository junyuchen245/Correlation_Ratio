import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch.nn as nn

class MIND_loss(torch.nn.Module):
    """
        Local (over window) normalized cross correlation loss.
        """

    def __init__(self, win=None):
        super(MIND_loss, self).__init__()
        self.win = win

    def pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img, radius=2, dilation=2):
        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

        # kernel size
        kernel_size = radius * 2 + 1

        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()

        # squared distances
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        rpad1 = nn.ReplicationPad3d(dilation)
        rpad2 = nn.ReplicationPad3d(radius)

        # compute patch-ssd
        ssd = F.avg_pool3d(rpad2(
            (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
                           kernel_size, stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind /= mind_var
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

        return mind

    def forward(self, y_pred, y_true):
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)

class MutualInformation(torch.nn.Module):
    """
    Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)

class localMutualInformation(torch.nn.Module):
    """
    Local Mutual Information for non-overlapping patches
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32, patch_size=5):
        super(localMutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 2 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
        self.patch_size = patch_size

    def local_mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """Making image paddings"""
        if len(list(y_pred.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (z_r // 2, z_r - z_r // 2, y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        elif len(list(y_pred.size())[2:]) == 2:
            ndim = 2
            x, y = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            padding = (y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        else:
            raise Exception('Supports 2D and 3D but not {}'.format(list(y_pred.size())))
        y_true = F.pad(y_true, padding, "constant", 0)
        y_pred = F.pad(y_pred, padding, "constant", 0)

        """Reshaping images into non-overlapping patches"""
        if ndim == 3:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 3, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 3, 1))
        else:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 3, 5)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 2, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 3, 5)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 2, 1))

        """Compute MI"""
        I_a_patch = torch.exp(- self.preterm * torch.square(y_true_patch - vbc))
        I_a_patch = I_a_patch / torch.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = torch.exp(- self.preterm * torch.square(y_pred_patch - vbc))
        I_b_patch = I_b_patch / torch.sum(I_b_patch, dim=-1, keepdim=True)

        pab = torch.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / self.patch_size ** ndim
        pa = torch.mean(I_a_patch, dim=1, keepdim=True)
        pb = torch.mean(I_b_patch, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()

    def forward(self, y_true, y_pred):
        return -self.local_mi(y_true, y_pred)
    

class CorrRatio(torch.nn.Module):
    """
    Correlation Ratio based on Parzen window
    Implemented by Junyu Chen, jchen245@jhmi.edu
    TODO: Under testing

    The Correlation Ratio as a New Similarity Measure for Multimodal Image Registration
    by Roche et al. 1998
    https://link.springer.com/chapter/10.1007/BFb0056301
    """

    def __init__(self, bins=32, sigma_ratio=1):
        super(CorrRatio, self).__init__()
        self.num_bins = bins
        bin_centers = np.linspace(0, 1, num=bins)
        self.vol_bin_centers = Variable(torch.linspace(0, 1, bins), requires_grad=False).cuda().view(1, 1, bins, 1)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 2 / (2 * sigma ** 2)

    def gaussian_kernel(self, diff, preterm):
        return torch.exp(- preterm * torch.square(diff))#torch.exp(-0.5 * (diff ** 2) / (sigma ** 2))

    def correlation_ratio(self, X, Y):
        B, C, H, W, D = Y.shape
        y_flat = Y.reshape(B, C, -1)  # Flatten spatial dimensions
        x_flat = X.reshape(B, C, -1)

        bins = self.vol_bin_centers

        # Calculate distances from each pixel to each bin
        y_expanded = y_flat.unsqueeze(2)  # [B, C, 1, H*W*D]
        diff = y_expanded - bins  # Broadcasted subtraction

        # Apply Parzen window approximation
        weights = self.gaussian_kernel(diff, preterm=self.preterm)
        weights_norm = weights / (torch.sum(weights, dim=-1, keepdim=True)+1e-5)
        # Compute weighted mean intensity in y_pred for each bin
        x_flat_expanded = x_flat.unsqueeze(2)  # Shape: [B, C, 1, H*W*D]
        mean_intensities = torch.sum(weights_norm * x_flat_expanded, dim=3)  # conditional mean, [B, C, bin]
        bin_counts = torch.sum(weights, dim=3)
        # mean_intensities = weighted_sums / (bin_counts + 1e-8)  # Add epsilon to avoid division by zero

        # Compute total mean of y_pred
        total_mean = torch.mean(x_flat, dim=2, keepdim=True) # [B, C, 1]

        # Between-group variance
        between_group_variance = torch.sum(bin_counts * (mean_intensities - total_mean) ** 2, dim=2) / (torch.sum(
            bin_counts, dim=2)+1e-5)

        # Total variance
        total_variance = torch.var(x_flat, dim=2)

        # Correlation ratio
        eta_square = between_group_variance / (total_variance + 1e-5)

        return eta_square.mean()/3

    def forward(self, y_true, y_pred):
        CR = self.correlation_ratio(y_true, y_pred) + self.correlation_ratio(y_pred, y_true)
        return -CR/2

class LocalCorrRatio(torch.nn.Module):
    """
    Localized Correlation Ratio based on Parzen window
    Implemented by Junyu Chen, jchen245@jhmi.edu
    TODO: Under testing

    The Correlation Ratio as a New Similarity Measure for Multimodal Image Registration
    by Roche et al. 1998
    https://link.springer.com/chapter/10.1007/BFb0056301
    """

    def __init__(self, bins=32, sigma_ratio=1, win=9):
        super(LocalCorrRatio, self).__init__()
        self.num_bins = bins
        bin_centers = np.linspace(0, 1, num=bins)
        self.vol_bin_centers = Variable(torch.linspace(0, 1, bins), requires_grad=False).cuda().view(1, 1, bins, 1)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 2 / (2 * sigma ** 2)
        self.win = win

    def gaussian_kernel(self, diff, preterm):
        return torch.exp(- preterm * torch.square(diff))

    def correlation_ratio(self, X, Y):
        B, C, H, W, D = Y.shape

        h_r = -H % self.win
        w_r = -W % self.win
        d_r = -D % self.win
        padding = (d_r // 2, d_r - d_r // 2, w_r // 2, w_r - w_r // 2, h_r // 2, h_r - h_r // 2, 0, 0, 0, 0)
        X = F.pad(X, padding, "constant", 0)
        Y = F.pad(Y, padding, "constant", 0)

        B, C, H, W, D = Y.shape
        num_patch = (H // self.win) * (W // self.win) * (D // self.win)
        x_patch = torch.reshape(X, (B, C, H // self.win, self.win, W // self.win, self.win, D // self.win, self.win))
        x_flat = x_patch.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B*num_patch, C, self.win ** 3)

        y_patch = torch.reshape(Y, (B, C, H // self.win, self.win, W // self.win, self.win, D // self.win, self.win))
        y_flat = y_patch.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B * num_patch, C, self.win ** 3)

        bins = self.vol_bin_centers

        # Calculate distances from each pixel to each bin
        y_expanded = y_flat.unsqueeze(2)  # [B*num_patch, C, 1, win**3]
        diff = y_expanded - bins  # Broadcasted subtraction

        # Apply Parzen window approximation
        weights = self.gaussian_kernel(diff, preterm=self.preterm)
        weights_norm = weights / (torch.sum(weights, dim=-1, keepdim=True)+1e-5)
        # Compute weighted mean intensity in y_pred for each bin
        x_flat_expanded = x_flat.unsqueeze(2)  # Shape: [B*num_patch, C, 1, win**3]
        mean_intensities = torch.sum(weights_norm * x_flat_expanded, dim=3)  # conditional mean, [B*num_patch, C, bin]
        bin_counts = torch.sum(weights, dim=3)
        # mean_intensities = weighted_sums / (bin_counts + 1e-8)  # Add epsilon to avoid division by zero

        # Compute total mean of y_pred
        total_mean = torch.mean(x_flat, dim=2, keepdim=True)  # [B*num_patch, C, 1]

        # Between-group variance
        between_group_variance = torch.sum(bin_counts * (mean_intensities - total_mean) ** 2, dim=2) / torch.sum(
            bin_counts, dim=2)

        # Total variance
        total_variance = torch.var(x_flat, dim=2)

        # Correlation ratio
        eta_square = between_group_variance / (total_variance + 1e-5)

        return eta_square.mean() / 3

    def forward(self, y_true, y_pred):
        CR = self.correlation_ratio(y_true, y_pred) + self.correlation_ratio(y_pred, y_true) #make it symmetric

        shift_size = self.win//2
        y_true = torch.roll(y_true, shifts=(-shift_size, -shift_size, -shift_size), dims=(2, 3, 4))
        y_pred = torch.roll(y_pred, shifts=(-shift_size, -shift_size, -shift_size), dims=(2, 3, 4))

        CR_shifted = self.correlation_ratio(y_true, y_pred) + self.correlation_ratio(y_pred, y_true)
        return -CR/4 - CR_shifted/4
    
class MI:
    """Mutual Information from TransMorph implementation"""

    @classmethod
    def loss(cls, pred, target, pred_mask=None, target_mask=None):
        """Create bin centers"""
        eps = 1e-8
        sigma_ratio = 1
        minval = 0.
        maxval = 1. # assume the intensity range of [0, 1]
        num_bin = 32

        vol_bin_centers = torch.linspace(minval, maxval, num_bin).to(pred.device)
        """Sigma for Gaussian approx."""
        sigma = torch.mean(torch.diff(vol_bin_centers)).item() * sigma_ratio
        preterm = 2 / (2 * sigma ** 2)

        pred = torch.clamp(pred, 0., maxval)
        target = torch.clamp(target, 0, maxval)

        target = target.view(target.shape[0], -1)
        pred = pred.view(pred.shape[0], -1)

        target = torch.unsqueeze(target, 2)
        pred = torch.unsqueeze(pred, 2)

        nb_voxels = pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        vbc = vol_bin_centers[None,None,...]

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- preterm * torch.square(target - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- preterm * torch.square(pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + eps
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + eps), dim=1), dim=1)
        return -mi.mean()  # average across batch