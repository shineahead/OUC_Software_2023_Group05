import io
import os
from utils import detection
from PIL import Image
import base64
import numpy as np
import torch

# 把base64的图片编码解析成PIL类型
def base64TwoPIL(base):
    img_data = base.split(",")[1]
    img_binary = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_binary))
    # img.show() # 展示图片
    return img

# 把PIL类型图片变成base64编码类型
def PILTwobase64(pil):
    # 将 PIL 图片对象转换为 BytesIO 对象
    buffer = io.BytesIO()
    pil.save(buffer, format="BMP")
    # 获取图像的base64编码
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return "data:image/bmp;base64," + img_base64

# 导入两张PIL类型的图片数据，out表示要保存的图片名称
def tool(img1, img2, out):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = detection.Net().eval().to(device)
    current_directory = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(current_directory, "model", "model.pth")
    model.load_state_dict(torch.load(path))

    # 输入图片
    windowSize = 7
    # 把图片
    im1, im2 = img1.convert('L'), img2.convert('L')
    # 这里把图片类型从PIL变成np.array
    im1, im2 = np.array(im1), np.array(im2)
    print(im1.shape)
    # shape维度变成了H、W、C
    im1 = im1.reshape(im1.shape[0], im1.shape[1], 1)
    im2 = im2.reshape(im2.shape[0], im2.shape[1], 1)

    height, width, c = im1.shape
    margin = (windowSize - 1) // 2
    im1 = detection.addZeroPadding(im1, margin=margin)
    im2 = detection.addZeroPadding(im2, margin=margin)

    outputs = np.zeros((height, width))
    print("检测开始-----")
    for i in range(height):
        for j in range(width):
            patch1 = im1[i:i + windowSize, j:j + windowSize, :]
            patch1 = patch1.reshape(1, patch1.shape[0], patch1.shape[1], patch1.shape[2])
            X_test_image = torch.FloatTensor(patch1.transpose(0, 3, 1, 2)).to(device)
            patch2 = im2[i:i + windowSize, j:j + windowSize, :]
            patch2 = patch2.reshape(1, patch2.shape[0], patch2.shape[1], patch2.shape[2])
            X_test_image1 = torch.FloatTensor(patch2.transpose(0, 3, 1, 2)).to(device)
            _, _, prediction = model(X_test_image, X_test_image1)
            prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
            outputs[i][j] = prediction
        if i % 20 == 0:
            print('... ... row ', i, ' handling ... ...')

    # postprocess
    outputs = detection.postprocess(outputs)
    # sio.savemat('result.mat', {'output': outputs})
    outputs = outputs * 255  # 把范围映射到0和255
    # 把np数组类型转化成PIL.Image类型
    image = Image.fromarray(outputs.astype('uint8'), 'L')  # 'L' 表示黑白模式
    image.save(os.path.join(os.path.dirname(current_directory), "img", out))

    print("检测完成-----")
    return image  # 返回PIL类型的图片数据
