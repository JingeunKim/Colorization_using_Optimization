import numpy as np
import matplotlib.pyplot as plt
from math import *
from Color import getColorExact, RGBtoYIQ, YIQtoRGB
import imageio

g_name = 'example.bmp'
c_name = 'example_marked.bmp'
out_name = 'example_res.bmp'

gI = imageio.imread(g_name) / 255.0
cI = imageio.imread(c_name) / 255.0


colorIm = (np.sum(abs(gI - cI), axis=2) > 0.01)
print(colorIm.shape)

sgI = RGBtoYIQ(gI)
scI = RGBtoYIQ(cI)

YUV = np.zeros_like(sgI)
YUV[:, :, 0] = sgI[:, :, 0] #Y
YUV[:, :, 1] = scI[:, :, 1] #I
YUV[:, :, 2] = scI[:, :, 2] #Q

max_d = int(floor(log(min(YUV.shape[0], YUV.shape[1])) / log(2) - 2))
iu = int(floor(YUV.shape[0] / (2 ** (max_d - 1))) * (2 ** (max_d - 1)))
ju = int(floor(YUV.shape[1] / (2 ** (max_d - 1))) * (2 ** (max_d - 1)))

id = 0
jd = 0
colorIm = colorIm[id:iu, jd:ju].copy()
print(colorIm.shape)
YUV = YUV[id:iu, jd:ju, :].copy()

nI = getColorExact(colorIm, YUV)
snI = nI
nI = YIQtoRGB(nI)

plt.imshow(nI)
plt.show()
plt.imsave(out_name, nI)
