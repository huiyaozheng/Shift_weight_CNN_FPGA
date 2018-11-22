# Copyright (C) 2018 Huiyao Zheng
# 
# This file is part of Shift_weight_CNN_FPGA.
# 
# Shift_weight_CNN_FPGA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Shift_weight_CNN_FPGA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Shift_weight_CNN_FPGA.  If not, see <http://www.gnu.org/licenses/>.





from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import math
import time


# Definition of the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x's shape is [1,20,4,4]
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def encode_as_fixed_point(value, int_bits, frac_bits):
    """
    Convert a floating point value into a (1+int_bits+frac_bits)-bit fixed point number.
    :param value: value to convert
    :param int_bits: number of bits representing the integer part
    :param frac_bits: number of bits representing the fractional part
    :return: converted value as a binary string
    """
    word_bits = int_bits + frac_bits
    is_neg = False
    if value < 0:
        is_neg = True
        value = -value

    # truncate the value if it exceeds the upper bound of representation
    max_exp = int(math.floor(math.sqrt(value)))
    for i in range(max_exp, int_bits - 1, -1):
        if value > math.pow(2, i):
            value = value - math.pow(2, i)

    # truncate the value if it exceeds the lower bound of representation
    min_exp = pow(2, -frac_bits)
    if value <= min_exp:
        fixed_point = ''
        for i in range(0, word_bits + 1):
            fixed_point = '0' + fixed_point
        return fixed_point

    # convert the value into string
    fixed_point = ''
    for i in range(0, word_bits):
        stride = math.pow(2, word_bits - frac_bits - i - 1)
        if value >= stride:
            value = value - stride
            fixed_point = fixed_point + '1'
        else:
            fixed_point = fixed_point + '0'

    # apply 2's complement if the value is negative
    if is_neg:
        fixed_point = int(fixed_point, 2) - (1 << (word_bits + 1))
        fixed_point = bin(fixed_point)[3:]
    else:
        fixed_point = '0' + fixed_point

    return fixed_point


def encode_as_shift(value):
    """
    Encode the value into 8-bit ABCCCCCC
    A = value < 0
    B = log2(abs(value)) < 0
    CCCCCC = abs(log2(abs(value))) in base 2
    :param value: value to convert
    :return: converted value as a binary string
    """
    ret = '1' if value < 0 else '0'
    log_value = math.log2(abs(value))
    ret += '1' if log_value < 0 else '0'
    powers = [32, 16, 8, 4, 2, 1]
    log_value = round(abs(log_value))
    for i in powers:
        if log_value >= i:
            ret += '1'
            log_value -= i
        else:
            ret += '0'
    return ret


def string_to_int(s):
    """
    Convert a binary string into a 2's complement signed integer
    :param s: binary string input
    :return: 2's complement signed integer
    """
    if s[0] == '0':
        return int(s, 2)
    else:
        length = len(s)
        return int(s[1:], 2) - pow(2, length - 1)


def print_tensor(tensor, transformation):
    """
    Coalesced print function able to handle 1D-4D tensors.
    :param tensor: Input tensor of dimension 1-4
    :param transformation: function to apply to each value, e.g. encoding as shift or fixed point
    """
    dimensions = tensor.shape
    dim = len(dimensions)
    if dim == 1:
        print('{', end='')
        count_i = 0
        for i in tensor:
            print(transformation(i), end='')
            count_i += 1
            if count_i < dimensions[0]:
                print(', ', end='')
        print('}')
    elif dim == 2:
        print('{')
        count_i = 0
        for i in tensor:
            print('{', end='')
            count_j = 0
            for j in i:
                print(transformation(j), end='')
                count_j += 1
                if count_j < dimensions[1]:
                    print(", ", end='')
            print('}', end='')
            count_i += 1
            if count_i < dimensions[0]:
                print(',')
        print('}')
    elif dim == 3:
        print('{')
        count_i = 0
        for i in tensor:
            print('{')
            count_j = 0
            for j in i:
                print('{', end='')
                count_k = 0
                for k in j:
                    print(transformation(k), end='')
                    count_k += 1
                    if count_k < dimensions[2]:
                        print(', ', end='')
                count_j += 1
                if count_j < dimensions[1]:
                    print('},')
            print('}', end='')
            count_i += 1
            if count_i < dimensions[0]:
                print('},')
        print('}')
        print('}')
    elif dim == 4:
        print('{')
        count_i = 0
        for i in tensor:
            print('{')
            count_j = 0
            for j in i:
                print('{', end='')
                count_k = 0
                for k in j:
                    print('{', end='')
                    count_l = 0
                    for l in k:
                        print(transformation(l), end='')
                        count_l += 1
                        if count_l < dimensions[3]:
                            print(', ', end='')
                    print('}', end='')
                    count_k += 1
                    if count_k < dimensions[2]:
                        print(', ', end='')
                count_j += 1
                if count_j < dimensions[1]:
                    print('},')
            print('}')
            count_i += 1
            if count_i < dimensions[0]:
                print('},')
        print('}')
        print('}')


def print_layer_1_weights(model):
    layer_1_para = model.state_dict().get('conv1.weight')
    print_tensor(layer_1_para, lambda x: int(encode_as_shift(x.item()), 2))
    print(layer_1_para.size())


def print_layer_1_bias(model):
    layer_1_bias = model.state_dict().get('conv1.bias')
    print_tensor(layer_1_bias, lambda x: string_to_int(encode_as_fixed_point(x.item(), 0, 7)))


def print_layer_2_weights(model):
    layer_2_para = model.state_dict().get('conv2.weight')
    print_tensor(layer_2_para, lambda x: int(encode_as_shift(x.item()), 2))
    print(layer_2_para.size())


def print_layer_2_bias(model):
    layer_2_bias = model.state_dict().get('conv2.bias')
    print_tensor(layer_2_bias, lambda x: string_to_int(encode_as_fixed_point(x.item() << 2, 0, 7)))


def print_layer_3_weights(model):
    layer_3_para = model.state_dict().get('fc1.weight')
    print_tensor(layer_3_para, lambda x: int(encode_as_shift(x.item()), 2))
    print(layer_3_para.size())


def print_layer_3_bias(model):
    layer_3_bias = model.state_dict().get('fc1.bias')
    print_tensor(layer_3_bias, lambda x: string_to_int(encode_as_fixed_point(x.item() << 2, 0, 7)))


def print_layer_4_weights(model):
    layer_4_para = model.state_dict().get('fc2.weight')
    print_tensor(layer_4_para, lambda x: int(encode_as_shift(x.item()), 2))
    print(layer_4_para.size())


def print_layer_4_bias(model):
    layer_4_bias = model.state_dict().get('fc2.bias')
    print_tensor(layer_4_bias, lambda x: string_to_int(encode_as_fixed_point(x.item(), 0, 7)))


def print_parameters(model):
    # shift
    print_layer_1_weights(model)

    # 8-bit fixed point S.(7)
    print_layer_1_bias(model)

    # shift
    print_layer_2_weights(model)

    # 8-bit fixed point S.00(7)
    print_layer_2_bias(model)

    # shift
    print_layer_3_weights(model)

    # 8-bit fixed point S.00(7)
    print_layer_3_bias(model)

    # shift
    print_layer_4_weights(model)

    # 8-bit fixed point S.(7)
    print_layer_4_bias(model)


# Run the model to evaluate the hooks
def evaluate_model(model):
    model.eval()
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True)
    device = torch.device("cpu")
    correct = 0
    with torch.no_grad():
        start = time.perf_counter()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        end = time.perf_counter()
        print(end - start)
    print(100. * correct / len(test_loader.dataset))
    print(len(test_loader.dataset) / (end - start))


trainedModel = Net()
trainedModel.load_state_dict(torch.load("mnist.pth", map_location=lambda storage, loc: storage))
print_parameters(trainedModel)
evaluate_model(trainedModel)
