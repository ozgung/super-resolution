from __future__ import print_function

from math import log10

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import DBPN
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set


'''
# DBPN network configurations from the original paper
SS:     T = 2, feature ext: conv(64), conv(18), reconst: conv(1) (grayscale)
S:      T = 2, feature ext: conv(128), conv(32), reconst: conv(1) (grayscale)
M:      T = 4, feature ext: conv(128), conv(32), reconst: conv(1) (grayscale)
L:      T = 6, feature ext: conv(128), conv(32), reconst: conv(1) (grayscale)
D-DBPN: T = 7, feature ext: conv(256), conv(64), reconst: conv(3) (rgb)
'''
ss_params       = dict(T = 2, n_0 = 64,  n_r = 18, ch=1)
s_params        = dict(T = 2, n_0 = 128, n_r = 32, ch=1)
m_params        = dict(T = 4, n_0 = 128, n_r = 32, ch=1)
l_params        = dict(T = 6, n_0 = 128, n_r = 32, ch=1)
d_dbpn_params   = dict(T = 7, n_0 = 256, n_r = 64, ch=3)

s_rgb_params        = dict(T = 2, n_0 = 128, n_r = 32, ch=3)

# model params
model_params = s_rgb_params

model = DBPN(**model_params)
print(model)

cuda = True

print('===> Loading datasets')
upscale_factor = 2
nthreads = 4
batchSize = 2

if model_params['ch'] == 1:
    convert_gray=True
elif model_params['ch'] == 3:
    convert_gray = False
else: raise Exception("Data loader channel format error")

train_set = get_training_set(upscale_factor, convert_gray=convert_gray)
test_set = get_test_set(upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=nthreads, batch_size=batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=nthreads, batch_size=batchSize, shuffle=False)

print('===> Datasets loaded')

criterion = nn.MSELoss()

if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters())


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda()
            target = target.cuda()

        prediction = model(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

nEpochs = 1
for epoch in range(1, nEpochs + 1):
    print("===> Train")
    train(epoch)
    print("===> Test")
    test()
    checkpoint(epoch)