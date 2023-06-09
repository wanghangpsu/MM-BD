from __future__ import absolute_import
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
from src.resnet import ResNet18
import copy



parser = argparse.ArgumentParser(description='UnivBM')
parser.add_argument('--model_dir', default='model0', help='model path')
parser.add_argument('--attack_dir', default='attack0', help='attack path')
#parser.add_argument('--data_path', '-d', required=True, help='data path')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'




class CleanNet(nn.Module):

    def __init__(self):
        super(CleanNet, self).__init__()
        model = ResNet18()
        model = model.to(device)
        model.load_state_dict(torch.load('./'+args.model_dir+'/model.pth'))
        model.eval()
        self.model = model
        self.clamp_w1 = torch.ones([64, 1, 1]).to(device) + 7.0
        self.clamp_w2 = torch.ones([64, 1, 1]).to(device) + 7.0
        self.clamp_w3 = torch.ones([128, 1, 1]).to(device) + 7.0
        self.clamp_w1.requires_grad = True
        self.clamp_w2.requires_grad = True
        self.clamp_w3.requires_grad = True
    def forward(self, x):

        out = F.relu(self.model.bn1(self.model.conv1(x)))
        out = torch.min(out, self.clamp_w1)
        out = self.model.layer1(out)
        out = torch.min(out, self.clamp_w2)
        out = self.model.layer2(out)
        out = torch.min(out, self.clamp_w3)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.model.linear(out)
        return out


network = CleanNet()
network.to(device)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
all_ind = []
for s in range(10):
    ind = [i for i, label in enumerate(trainset.targets) if label == s]
    all_ind += ind[:50]
trainset = torch.utils.data.Subset(trainset, all_ind)
correct_idx = []
for i in range(trainset.__len__()):
    image, label = trainset.__getitem__(i)
    image = image.to(device).unsqueeze(0)
    out = network.model(image)
    _, predicted = out.max(1)
    if predicted.item() == label:
        correct_idx.append(i)

trainset = torch.utils.data.Subset(trainset, correct_idx)




trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=200, shuffle=True, num_workers=2)

optimizer = torch.optim.Adam( [network.clamp_w1, network.clamp_w2, network.clamp_w3], lr=0.1)
criterion = nn.CrossEntropyLoss()
mse = nn.MSELoss()
c = 0.5
a = 1.2
for epoch in range(50):
    correct = 0
    total = 0
    for idx, (images, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        ref_out = network.model(images)
        outputs = network(images)
        loss1 = mse(outputs, ref_out)
        loss2 = torch.norm(network.clamp_w1) \
                + torch.norm(network.clamp_w2) \
                + torch.norm(network.clamp_w3)
        #print(network.clamp_w1)
        loss = loss1 + c * loss2
        loss.backward()
        optimizer.step()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    if epoch > 10 and epoch % 10 == 0:
        if acc >= 95:
            c *= a
        else:
            c /= a


    #print('training acc rate: %.3f' % acc)




print('c: ' + str(c))




print(args.model_dir)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
all_ind = []
for s in range(10):
    ind = [i for i, label in enumerate(testset.targets) if label == s]
    all_ind += ind[50:]
testset = torch.utils.data.Subset(testset, all_ind)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = network(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

acc = 100.*correct/total
print('Test ACC: %.3f' % acc)




test_attacks = torch.load('./'+args.attack_dir + '/test_attacks')
test_images_attacks = test_attacks['image']
test_labels_attacks = test_attacks['label']
'''
# Normalize backdoor test images
for i in range(len(test_images_attacks)):
        est_images_attacks[i] = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(test_images_attacks[i])
'''
testset_attacks = torch.utils.data.TensorDataset(test_images_attacks, test_labels_attacks)
attackloader = torch.utils.data.DataLoader(testset_attacks, batch_size=100, shuffle=False, num_workers=2)

# Evaluate attack success rate
correct = 0
total = 0
corr = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(attackloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = network(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        #corr += predicted.eq(targets+1).sum().item()

acc = 100.*correct/total
print('Attack success rate: %.3f' % acc)


