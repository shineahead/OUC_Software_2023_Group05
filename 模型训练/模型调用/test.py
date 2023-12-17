import numpy as np
from skimage import io

im1 = io.imread("bern_2.bmp")[:, :, 0].astype(np.float32)
print(im1.shape)

