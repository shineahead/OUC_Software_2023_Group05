# coding = utf - 8
import scipy.io as sio
from PIL import Image

outputs = sio.loadmat("./data/data_1.mat")['data']
outputs = outputs * 255  # 把范围映射到0和255
print(type(outputs))

# 保存图片，将数组转换为 PIL 图像
image = Image.fromarray(outputs.astype('uint8'), 'L')  # 'L' 表示黑白模式
image.save('./img/data_1.bmp')
