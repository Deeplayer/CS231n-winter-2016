import numpy as np
from scipy.misc import imread, imsave, imresize
import matplotlib.pyplot as plt

img = imread('puppy.jpg')
img1 = img.transpose(2, 0, 1)
x = np.zeros(img.shape)
print img1.shape
plt.imshow(img)
plt.show()

for i in xrange(517):
    for j in xrange(517):
        x[i, 517-1-j] = img[i, j]

plt.imshow(x.astype('uint8'))
plt.show()

ss = np.zeros(4)
dd = np.ones(3)
z = np.concatenate((ss, dd))
print z
