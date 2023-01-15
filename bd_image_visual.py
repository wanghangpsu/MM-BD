from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import random
import sys
import numpy as np

if not os.path.isdir('attack1'):
    print ('Attack images not found, please craft attack images first!')
    sys.exit(0)
train_attacks = torch.load('./example/train_attacks')
train_images_attacks = train_attacks['image']
train_labels_attacks = train_attacks['label']
test_attacks = torch.load('./example/test_attacks')
test_images_attacks = test_attacks['image']
test_labels_attacks = test_attacks['label']
pattern = torch.load('./example/pattern')
image = train_images_attacks[1]
image = image.numpy()
image = np.transpose(image, [1, 2, 0])
plt.imshow(image)
plt.show()
image = Image.fromarray(np.uint8(image*255))
pattern = pattern.numpy()
pattern = np.transpose(pattern, [1, 2, 0])
pattern = Image.fromarray(np.uint8(pattern*255))
pattern.save('./example/pattern.png')
image.save('./example/image.png')
