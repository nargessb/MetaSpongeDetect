# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:14:36 2024

@author: narge
"""


import os, sys
from os.path import abspath

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import warnings
warnings.filterwarnings('ignore')


import torch
import torch.nn as nn

from art import config
from art.utils import load_dataset, get_file
from art.estimators.classification import PyTorchClassifier
from art.attacks.poisoning import FeatureCollisionAttack

import numpy as np

#%matplotlib inline
import matplotlib.pyplot as plt

np.random.seed(301)
#%%%%%%%%%%%
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('cifar10')

print(x_train.shape)
x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
# num_samples_train = 1000
# num_samples_test = 1000
# x_train = x_train[0:num_samples_train]
# y_train = y_train[0:num_samples_train]
# x_test = x_test[0:num_samples_test]
# y_test = y_test[0:num_samples_test]

class_descr = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print("shape of x_train",x_train.shape)
print("shape of y_train",y_train.shape)
#%%%%%%%%% Load Model to be Attacked
# Model Definition and pretrained model pulled from: 
# https://github.com/huyvnphan/PyTorch_CIFAR10
import torch
import torch.nn as nn
import os

__all__ = ["ResNet",
    "resnet18"]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, device, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        # Download the model state_dict from the link: and run your code
        state_dict = torch.load(
            'resnet18.pt', map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, **kwargs
    )

import torch.optim as optim
# Pretrained model
from torchvision import models
#%%% get classifiers
import copy

# Assume `model` is the original PyTorch model, and `classifier` is a PyTorchClassifier instance
original_model = models.resnet18(pretrained=True)  # Recreate the original model setup
num_ftrs = original_model.fc.in_features
original_model.fc = nn.Linear(num_ftrs, 10)  # Adjust output to 10 classes for CIFAR-10

# Clone the model for poisoned training
poisoned_model = copy.deepcopy(original_model)

# Create two classifier instances, one for each model
clean_classifier = PyTorchClassifier(
    model=original_model, 
    clip_values=(min_, max_),
    preprocessing=((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    nb_classes=10,
    input_shape=(3, 32, 32),
    loss=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(original_model.parameters(), lr=0.0001)
)

poisoned_classifier = PyTorchClassifier(
    model=poisoned_model,
    clip_values=(min_, max_),
    preprocessing=((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    nb_classes=10,
    input_shape=(3, 32, 32),
    loss=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(poisoned_model.parameters(), lr=0.0001)
)
# Train on clean data
clean_classifier.fit(x_train, y_train, nb_epochs=20, batch_size=256)
def evaluate_model(classifier, x_test, y_test, class_descr):
    # Predict using the classifier
    predictions = classifier.predict(x_test)
    # Calculate accuracy
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

# Call the evaluate_model function with the classifier and datasets
print("Evaluating clean model:")
evaluate_model(clean_classifier, x_test, y_test, class_descr)
#%%%% Get Target
class_descr = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

target_class = "automobile"
# Example: Let's assume you know the index of the class from some prior step
class_index = class_descr.index(target_class)  # Correct as per your Out[53]
class_indices = np.argmax(y_test, axis=1)  # This converts each one-hot vector to the index of the 'hot' (i.e., 1) position
# Filter x_test to get only instances of the target class
is_target_class = class_indices == class_index  # This creates a boolean array where True indicates the target class
target_class_instances = x_test[is_target_class]
print("Shape of target class instances:", target_class_instances.shape)

if target_class_instances.shape[0] > 3:
    target_instance = np.expand_dims(target_class_instances[3], axis=0)
    print("Shape of the selected target instance:", target_instance.shape)
else:
    print("Not enough instances of the target class available.")  
target_label = np.zeros(len(class_descr))
target_label[class_descr.index(target_class)] = 1

feature_layer = poisoned_classifier.layer_names[-2]
print(feature_layer)

base_class = "airplane"

base_class_index = class_descr.index(base_class)  # Correct as per your Out[53]

is_base_class = class_indices == base_class_index  # This creates a boolean array where True indicates the target class
base_class_instances = x_test[is_base_class]
print("Shape of target class instances:", base_class_instances.shape)

if base_class_instances.shape[0] > 3:
    # Select the fourth instance (index 3) of the target class
    base_class_instance = np.expand_dims(base_class_instances[3], axis=0)
    print("Shape of the selected target instance:", base_class_instance.shape)
else:
    print("Not enough instances of the target class available.")  
base_class_label = np.zeros(len(class_descr))
base_class_label[class_descr.index(base_class)] = 1

x_test_pred = np.argmax(poisoned_classifier.predict(base_class_instances), axis=1)
nb_correct_pred = np.sum(x_test_pred == np.argmax(base_class_label, axis=0))

attack = FeatureCollisionAttack(poisoned_classifier, 
                                target_instance, 
                                feature_layer, 
                                max_iter=10, 
                                similarity_coeff=256,
                                watermark=0.3,
                                learning_rate=1)
poison_new, poison_labels_new = attack.poison(base_class_instances)
poison_pred = np.argmax(poisoned_classifier.predict(poison_new), axis=1)
plt.figure(figsize=(10,10))


