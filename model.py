import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# class Net(nn.Module):
#     def __init__(self, upscale_factor):
#         super(Net, self).__init__()
#
#         self.relu = nn.ReLU()
#         self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
#         self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
#         self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
#         self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
#
#         self._initialize_weights()
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = self.pixel_shuffle(self.conv4(x))
#         return x
#
#     def _initialize_weights(self):
#         init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
#         init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
#         init.orthogonal(self.conv3.weight, init.calculate_gain('relu'))
#         init.orthogonal(self.conv4.weight)

class DBPN(nn.Module):
    def __init__(self, T, n_0, n_r, ):
        super(DBPN, self).__init__()

        self.ml = nn.ModuleList()

        self.H_list = []

        # Inital feature extraction (sec. 3.3.1)
        self.ml.append(nn.Conv2d(3, n_0, 3)) # N,3,H,W -> N,n_0,H,W
        self.ml.append(nn.Conv2d(n_0, n_r, 1)) # N,n_0,H,W -> N,n_r,H,W

        # Back projection stages (sec. 3.3.2)
        for stage in range(T-1):
            self.ml.append(UpProjectionUnit(n_r))
            self.ml.append(DownProjectionUnit(n_r))
        self.ml.append(UpProjectionUnit(n_r))

        # Reconstruction (sec. 3.3.3)
        # TODO: concat T tensors
        self.ml.append(nn.Conv2d(T*n_r, 3, 3)) # N, T*n_r,H,W -> N,3,H,W


    def forward(self, x):
        for layer in self.ml:
            x=layer(x)
        return x


class UpProjectionUnit(nn.Module):
    def __init__(self, n_r, s=2, kernel_size=6, pad=2):
        super(UpProjectionUnit, self).__init__()

        self.deconv0 = nn.ConvTranspose2d(n_r, n_r, s, pad)
        self.conv0 = nn.Conv2d(n_r, n_r, s, pad)
        self.deconv1 = nn.ConvTranspose2d(n_r, n_r, s, pad)

        self.H_out = None

    def forward(self, x):
        L_prev = x
        H_0 = F.prelu(self.deconv0(x))
        L_0 = F.prelu(self.conv0(H_0))
        residual = L_0 - L_prev
        H_1 = F.prelu(self.deconv1(residual))
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

    def forward(self, x):
        H = x
        L_0 = F.prelu(self.conv0(x))
        H_0 = F.prelu(self.deconv0(x))
        residual = H_0 - H
        L_1 = F.prelu(self.conv1(residual))
        x = L_0 + L_1
        self.L_out = x
        return x
