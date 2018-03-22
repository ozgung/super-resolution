from __future__ import print_function

import numpy as np
import torch
import matplotlib.pyplot as plt
from model import DBPN


# model params
model_params = dict(
T = 3,
n_0 = 5,
n_r = 5 )

model = DBPN(**model_params)
print(model)

input = torch.rand((256,256))
plt.imshow(input.numpy())
plt.ioff()
plt.show()