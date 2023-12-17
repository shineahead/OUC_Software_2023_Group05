import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt

outputs = sio.loadmat("./result.mat")['output']
outputs = outputs * 255  # 把范围映射到0和255
print(type(outputs))

# 显示黑白图像
plt.imshow(outputs, cmap='gray')
plt.axis('off')  # 不显示坐标轴
plt.show()

# 保存图片，将数组转换为 PIL 图像
image = Image.fromarray(outputs.astype('uint8'), 'L')  # 'L' 表示黑白模式
image.save('result.png')
