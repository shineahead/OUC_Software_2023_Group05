# coding = utf-8
import os
import cv2
from functools import reduce

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage import io
from skimage import measure

'''
使用保存的pth模型来得出训练结果
'''
class FeatNet(nn.Module):
    def __init__(self):
        super(FeatNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(16)

        self.conv1_1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.conv1_2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.bn1_2 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 1, 2, 1)

        self.conv2_1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn2_2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 1, 2, 1)

        self.conv3_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn3_2 = nn.BatchNorm2d(64)

        # Feature fusion
        self.conv_fusion1 = nn.Conv2d(16, 64, 1, 4, 2)
        self.conv_fusion2 = nn.Conv2d(32, 64, 1, 2, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x = F.relu(self.bn1_1(self.conv1_1(x1)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x)))
        x = x1 + x_1
        x2 = self.conv2(x)
        x = F.relu(self.bn2_1(self.conv2_1(x2)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x)))
        x = x2 + x_2
        x3 = self.conv3(x)
        x = F.relu(self.bn3_1(self.conv3_1(x3)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x)))
        return x_1, x_2, x_3

class FeatFuse(nn.Module):
    def __init__(self):
        super(FeatFuse, self).__init__()

        self.conv_fusion1 = nn.Conv2d(16, 64, 1, 4, 3)
        self.conv_fusion2 = nn.Conv2d(32, 64, 1, 2, 1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 8)
        self.fc2 = nn.Linear(8, 64 * 3)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3):
        batch_size = x1.size(0)
        out_channels = x3.size(1)
        x1 = self.conv_fusion1(x1)
        x2 = self.conv_fusion2(x2)
        output = []
        output.append(x1)
        output.append(x2)
        output.append(x3)
        x = x1 + x2 + x3

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        a_b = x.reshape(batch_size, 3, out_channels, -1)
        a_b = self.softmax(a_b)
        # the part of selection
        a_b = list(a_b.chunk(3, dim=1))  # split to a and b
        a_b = list(map(lambda x: x.reshape(batch_size, out_channels, 1, 1), a_b))
        V = list(map(lambda x, y: x * y, output, a_b))
        V = reduce(lambda x, y: x + y, V)
        return V

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.featnet = FeatNet()
        self.featfuse = FeatFuse()
        self.featnet1 = FeatNet()
        self.featfuse1 = FeatFuse()
        # self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 2)

        self.global_pool1 = nn.AdaptiveAvgPool2d(1)
        self.global_pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 2)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x, y):
        x1_1, x1_2, x1_3 = self.featnet(x)
        x2_1, x2_2, x2_3 = self.featnet1(y)

        feat_11 = self.featfuse(x1_1, x1_2, x1_3)
        feat_22 = self.featfuse1(x2_1, x2_2, x2_3)
        feat_1 = self.global_pool1(feat_11)
        feat_2 = self.global_pool2(feat_22)
        feat_1 = feat_1.view(feat_1.size(0), -1)
        feat_2 = feat_2.view(feat_2.size(0), -1)
        feat_1 = self.fc1(feat_1)
        feat_2 = self.fc2(feat_2)

        feature_corr = self.xcorr_depthwise(feat_11, feat_22)
        feat = feature_corr.view(feature_corr.size(0), -1)
        # feat = global_pool(feature_corr)
        feat = self.fc(feat)
        return feat_1, feat_2, feat

    def xcorr_depthwise(self, x, kernel):
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out

def addZeroPadding(X, margin=2):
    newX = np.zeros((
        X.shape[0] + 2 * margin,
        X.shape[1] + 2 * margin,
        X.shape[2]
              ))
    newX[margin:X.shape[0]+margin, margin:X.shape[1]+margin, :] = X
    return newX

def postprocess(res):
    res_new = res
    res = measure.label(res, connectivity=2)
    num = res.max()
    for i in range(1, num+1):
        idy, idx = np.where(res == i)
        # 去除小噪声区域，当区域内的连通数量小于20
        if len(idy) <= 20:
            res_new[idy, idx] = 0
    return res_new

# 输入mat文件的路径，然后读取
def detect(origin="./data", path1="data_1.mat", path2="data_2.mat", out="result.bmp"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net().eval().to(device)
    model.load_state_dict(torch.load('./model/model.pth'))

    # 输入图片
    windowSize = 7
    # im1 = sio.loadmat(os.path.join(origin, path1))['data']
    # im2 = sio.loadmat(os.path.join(origin, path2))['data']
    # 这里读取的图片类型是np.array
    im1 = io.imread(os.path.join(origin, path1))[:, :, 0].astype(np.float32)
    im2 = io.imread(os.path.join(origin, path2))[:, :, 0].astype(np.float32)
    # shape维度变成了H、W、C
    im1 = im1.reshape(im1.shape[0], im1.shape[1], 1)
    im2 = im2.reshape(im2.shape[0], im2.shape[1], 1)

    height, width, c = im1.shape
    margin = (windowSize - 1) // 2
    im1 = addZeroPadding(im1, margin=margin)
    im2 = addZeroPadding(im2, margin=margin)

    outputs = np.zeros((height, width))
    print("开始写入-----")
    # 写入到.mat文件
    for i in range(height):
        for j in range(width):
            patch1 = im1[i:i+windowSize, j:j+windowSize, :]
            patch1 = patch1.reshape(1, patch1.shape[0], patch1.shape[1], patch1.shape[2])
            X_test_image = torch.FloatTensor(patch1.transpose(0, 3, 1, 2)).to(device)
            patch2 = im2[i:i+windowSize, j:j+windowSize, :]
            patch2 = patch2.reshape(1, patch2.shape[0], patch2.shape[1], patch2.shape[2])
            X_test_image1 = torch.FloatTensor(patch2.transpose(0, 3, 1, 2)).to(device)
            _, _, prediction = model(X_test_image, X_test_image1)
            prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
            outputs[i][j] = prediction
        if i % 20 == 0:
            print('... ... row ', i, ' handling ... ...')

    # postprocess
    outputs = postprocess(outputs)
    # sio.savemat('result.mat', {'output': outputs})
    outputs = outputs * 255  # 把范围映射到0和255
    # 把np数组类型转化成PIL.Image类型
    image = Image.fromarray(outputs.astype('uint8'), 'L')  # 'L' 表示黑白模式
    image.save(os.path.join("./img", out))

    print("写入完成-----")

if __name__ == "__main__":
    detect(origin="./SAR_train_data", path1="bern_1.bmp", path2="bern_2.bmp", out="bern_gt.bmp")
