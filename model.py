from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class DBPN(nn.Module):
    def __init__(self, T, n_0, n_r, ch=3):
        super(DBPN, self).__init__()

        self.ml = nn.ModuleList()

        self.H_list = []

        # Inital feature extraction (sec. 3.3.1)
        self.ml.append(nn.Conv2d(ch, n_0, 3, padding=1)) # N,ch,H,W -> N,n_0,H,W
        self.ml.append(nn.Conv2d(n_0, n_r, 1)) # N,n_0,H,W -> N,n_r,H,W

        # Back projection stages (sec. 3.3.2)
        for stage in range(T-1):
            self.ml.append(UpProjectionUnit(n_r))
            self.ml.append(DownProjectionUnit(n_r))
        self.ml.append(UpProjectionUnit(n_r))

        # Reconstruction (sec. 3.3.3)
        self.ml.append(nn.Conv2d(T*n_r, ch, 3, padding=1)) # N, T*n_r,H,W -> N,ch,H,W

        self.T = T
        self.n_0 = n_0
        self.n_r = n_r
        self.ch = ch

    def forward(self, x):
        # Feature Extraction layers
        print("input, ", x.data.shape)
        x = self.ml[0](x) # 32x3x28x28 -> 32x128x126x126
        x = self.ml[1](x) # 32x128x126x126 ->
        print("reconstruction input, ", x.data.shape)
        # reconstruction layerls
        i=2
        self.H_list = []
        for stage in range(self.T-1):
            x = self.ml[i](x) # upprojection
            self.H_list.append(x)
            x =  self.ml[i+1](x)# downprojection
            i += 2
        x = self.ml[i](x) # last upprojection
        self.H_list.append(x)
        print("activation output saved: ", x.data.shape)

        # reconstruction layer
        # concat activations in H_list
        x = torch.cat(self.H_list, dim=1)
        print("concatenated activations output, ", x.data.shape)
        x = self.ml[-1](x)
        print("output, ", x.data.shape)
        return x


class UpProjectionUnit(nn.Module):
    def __init__(self, n_r, s=2, kernel_size=6, pad=2):
        super(UpProjectionUnit, self).__init__()

        self.deconv0 = nn.ConvTranspose2d(n_r, n_r, s, pad)
        self.conv0 = nn.Conv2d(n_r, n_r, s, pad)
        self.deconv1 = nn.ConvTranspose2d(n_r, n_r, s, pad)

        self.H_out = None

        self.prelu0 = nn.PReLU()
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

    def forward(self, x):
        L_prev = x
        H_0 = self.prelu0(self.deconv0(x))
        L_0 = self.prelu1(self.conv0(H_0))
        residual = L_0 - L_prev
        H_1 = self.prelu2(self.deconv1(residual))
        x = H_0 + H_1
        self.H_out = x
        return x

class DownProjectionUnit(nn.Module):
    def __init__(self, n_r, s=2, kernel_size=6, pad=2):
        super(DownProjectionUnit, self).__init__()

        self.conv0 = nn.Conv2d(n_r, n_r, s, pad)
        self.deconv0 = nn.ConvTranspose2d(n_r, n_r, s, pad)
        self.conv1 = nn.Conv2d(n_r, n_r, s, pad)

        self.L_out = None

        self.prelu0 = nn.PReLU()
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

    def forward(self, x):
        H = x
        L_0 = self.prelu0(self.conv0(x))
        H_0 = self.prelu1(self.deconv0(L_0))
        residual = H_0 - H
        L_1 = self.prelu2(self.conv1(residual))
        x = L_0 + L_1
        self.L_out = x
        return x
