#!/usr/bin/env python
# coding: utf-8

# # ResNET50 Implementation
#
# Code was inspired by the official pytorch library:
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py


# Execute this code block to install dependencies when running on colab
try:
    import torch
except:
    from os.path import exists
    from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
    platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
    cuda_output = get_ipython().getoutput("ldconfig -p|grep cudart.so|sed -e 's/.*\\.\\([0-9]*\\)\\.\\([0-9]*\\)$/cu\\1\\2/'")
    accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

    get_ipython().system('pip install -q http://download.pytorch.org/whl/{accelerator}/torch-1.0.0-{platform}-linux_x86_64.whl torchvision')

try:
    import torchbearer
except:
    get_ipython().system('pip install torchbearer')


import torch
import torch.nn.functional as F
from torchsummary import summary

from torch import nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import FashionMNIST

import matplotlib.pyplot as plt
from torchbearer import Trial
import torchbearer
import torchvision.models as models


import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, interim_class=-1):
        self.interim_class = interim_class
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.avgpool0 = nn.AvgPool2d(1, stride=1)
        self.class0 = nn.Linear(3136,10)

        self.avgpool1 = nn.AvgPool2d(1, stride=1)
        self.class1 = nn.Linear(3136,10)

        self.avgpool2 = nn.AvgPool2d(1, stride=1)
        self.class2 = nn.Linear(2048,10)

        self.avgpool3 = nn.AvgPool2d(1, stride=1)
        self.class3 = nn.Linear(1024,10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        outputs=[]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if(self.interim_class == 0):
          out = torch.tensor(x)
          out = self.avgpool0(out)
          outputs.append(self.class0(out.view(out.size(0), -1)))

        x = self.layer1(x)
        if(self.interim_class == 1):
          out = torch.tensor(x)
          out = self.avgpool1(out)
          outputs.append(self.class1(out.view(out.size(0), -1)))

        x = self.layer2(x)
        if(self.interim_class == 2):
          out = torch.tensor(x)
          out = self.avgpool2(out)
          outputs.append(self.class2(out.view(out.size(0), -1)))
        x = self.layer3(x)
        if(self.interim_class == 3):
          out = torch.tensor(x)
          out = self.avgpool3(out)
          outputs.append(self.class3(out.view(out.size(0), -1)))

        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        outputs.append(x)
        return outputs


def ResNet18_0():
    return ResNet(BasicBlock, [2,2,2,2], interim_class=0)

def ResNet18_1():
    return ResNet(BasicBlock, [2,2,2,2], interim_class=1)

def ResNet18_2():
    return ResNet(BasicBlock, [2,2,2,2], interim_class=2)

def ResNet18_3():
    return ResNet(BasicBlock, [2,2,2,2], interim_class=3)



############# CREATE DATA ###################
import random
from torch.utils.data.sampler import SubsetRandomSampler

def train_val_sampler(trainset, valset, split):
  """ Return a sampler for the training and validation sets. This method is inspired by
      https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb """
  random.seed(10)
  size = len(trainset)
  indexes = list(range(size))
  random.shuffle(indexes)

  train_indexes = indexes[0:split]
  val_indexes = indexes[split:]

  train_sampler = SubsetRandomSampler(train_indexes)
  val_sampler = SubsetRandomSampler(val_indexes)

  return train_sampler, val_sampler

# This normalise transform was taken from https://github.com/kalviny/MSDNet-PyTorch/blob/master/dataloader.py

normalise = transforms.Normalize(mean=[0.5],
                                 std=[0.5])

train_transform = transforms.Compose([  # convert to tensor
    transforms.ToTensor(),
])

test_val_transform = transforms.Compose([
      transforms.ToTensor(),  # convert to tensor
])

# load data
trainset = FashionMNIST(".", train=True, download=True, transform=train_transform)
valset = FashionMNIST(".", train=True, download=True, transform=test_val_transform)
testset = FashionMNIST(".", train=False, download=True, transform=test_val_transform)

print(len(testset))
# get the samplers
train_sampler, val_sampler = train_val_sampler(trainset, valset, 50000)

# create data loaders
trainloader = DataLoader(trainset, batch_size=128, sampler = train_sampler)
valloader = DataLoader(valset, batch_size=128, sampler = val_sampler)
testloader = DataLoader(testset, batch_size=1, shuffle=True)


def evaluate_model(model, test_loader, evaluation_type='all'):
  """
  Evaluate the model on the validation set.
  """
  # model.eval()
  # model.to(device)
  model.evaluation_type = 'all'
  num_classifiers = 2
  correct = [0 for i in range(num_classifiers)]
  total = [0 for i in range(num_classifiers)]

  with torch.no_grad():
      testloader = tqdm(test_loader)
      for data in testloader:
          images, labels = data
          images = images.to(device)
          labels = labels.to(device)
          outputs = model(images)

          for i in range(len(correct)):
            classifier = outputs[i]
            _, predicted = torch.max(classifier.data, 1)

            total[i] += labels.size(0)
            correct[i] += (predicted == labels).sum().item()

  correct = [100 * correct[i] / total[i] for i in range(len(total))]
  return correct

def evaluate_final_classifier(model, trainloader):
  model.eval()
  model.to(device)
  correct = 0
  total = 0

  with torch.no_grad():
    train_loader = tqdm(trainloader)
    for data in train_loader:
      images, labels = data
      images = images.to(device)
      labels = labels.to(device)
      outputs = F.softmax(model(images)[-1]) # get the output of the last classifier

      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      accuracy = 100 * correct / total

      train_loader.set_postfix(accuracy=accuracy)

  print(correct)
  return accuracy

class AverageCrossEntropyLoss(nn.Module):
  def __init__(self):
    super(AverageCrossEntropyLoss, self).__init__()

  def forward(self, outputs, labels):

    total_loss = 0

    for i in range(len(outputs)):
      output = outputs[i]
      loss = F.cross_entropy(output, labels)
      total_loss += loss
    return total_loss / len(outputs)

from tqdm.auto import tqdm
import copy
import pandas as pd

model = ResNet18_2()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

epoch_range = 30
csv_data = pd.DataFrame(columns=['epoch', 'lr', 'train_acc', 'val_acc','average_train_acc', 'average_val_acc'])


crit = AverageCrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=[4,24], gamma=0.1)
best_acc = 0

for epoch in range(epoch_range):  # loop over the dataset multiple times
    train_loader = tqdm(trainloader)
    running_loss = 0.0

    for param_group in optimiser.param_groups:
        lr = param_group['lr']

    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimiser.zero_grad()


        outputs = model(inputs)
        loss = crit(outputs, labels)
        loss.backward()
        optimiser.step()

        # print statistics
        running_loss += loss.item()
        train_loader.set_postfix(epoch=epoch, lr = lr)
    scheduler.step()

    train_acc = evaluate_model(model, trainloader)
    val_acc = evaluate_model(model, valloader)
    average_train_acc = sum(train_acc) / len(train_acc)
    average_val_acc = sum(val_acc) / len(val_acc)

    print("train_acc", train_acc)
    print("val_acc", val_acc)
    print("average_train_acc", average_train_acc)
    print("average_val_acc", average_val_acc)

    # update the best model
    if average_val_acc > best_acc:
      best_model = copy.deepcopy(model)
      best_acc = average_val_acc
      best_epoch = epoch

    results = {
        'epoch' : epoch,
        'lr' : lr,
        'train_acc' : train_acc,
        'val_acc' : val_acc,
        'average_train_acc' : average_train_acc,
        'average_val_acc' : average_val_acc
    }

    csv_data = csv_data.append(results, ignore_index=True)

best_data = pd.DataFrame({
    'best_epoch' : [best_epoch],
    'best_val_acc' : [best_acc]
})

print('Finished Training')


train_acc = evaluate_model(model, trainloader)
print(train_acc)


val_acc = evaluate_model(model, valloader)
print(val_acc)
