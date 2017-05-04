import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from pretrained_cnn import PretrainedCNN
from data_utils import load_tiny_imagenet
from image_utils import blur_image, deprocess_image, preprocess_image
from tqdm import *

data = load_tiny_imagenet('E:/PycharmProjects/ML/CS231n/RNN/datasets/tiny-imagenet-100-A', subtract_mean=True)
model = PretrainedCNN(h5_file='E:/PycharmProjects/ML/CS231n/RNN/datasets/pretrained_model.h5')

def deepdream(X, layer, model, **kwargs):
    """
    Generate a DeepDream image.

    Inputs:
    - X: Starting image, of shape (1, 3, H, W)
    - layer: Index of layer at which to dream
    - model: A PretrainedCNN object

    Keyword arguments:
    - learning_rate: How much to update the image at each iteration
    - max_jitter: Maximum number of pixels for jitter regularization
    - num_iterations: How many iterations to run for
    - show_every: How often to show the generated image
    """

    X = X.copy()
    learning_rate = kwargs.pop('learning_rate', 5.0)
    max_jitter = kwargs.pop('max_jitter', 16)
    num_iterations = kwargs.pop('num_iterations', 200)
    show_every = kwargs.pop('show_every', 50)

    for t in tqdm(xrange(num_iterations)):
        # As a regularizer, add random jitter to the image
        ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
        X = np.roll(np.roll(X, ox, -1), oy, -2)

        # Forward until dreaming layer
        fea, cache = model.forward(X, end=layer ,mode='test')

        # Set the gradient equal to the feature
        dfea = fea
        dX, grads = model.backward(dfea, cache)

        # Step of gradient descent
        X += learning_rate*dX

        # Undo the jitter
        X = np.roll(np.roll(X, -ox, -1), -oy, -2)

        # As a regularizer, clip the image
        mean_pixel = data['mean_image'].mean(axis=(1, 2), keepdims=True)
        X = np.clip(X, -mean_pixel, 255.0 - mean_pixel)

        # Periodically show the image
        if t == 0 or (t + 1) % show_every == 0:
            img = deprocess_image(X, data['mean_image'], mean='pixel')
            plt.imshow(img)
            plt.title('Iteration: %d' % (t + 1))
            plt.gcf().set_size_inches(8, 8)
            plt.axis('off')
            plt.show()

    return X


def read_image(filename, max_size):
  """
  Read an image from disk and resize it so its larger side is max_size
  """
  img = imread(filename)
  H, W, _ = img.shape
  if H >= W:
      img = imresize(img, (max_size, int(W * float(max_size) / H)))
  elif H < W:
      img = imresize(img, (int(H * float(max_size) / W), max_size))
  return img

filename = 'leaning tower.jpg'
max_size = 1024
img = read_image(filename, max_size)
plt.imshow(img)
plt.axis('off')

# Preprocess the image by converting to float, transposing,
# and performing mean subtraction.
img_pre = preprocess_image(img, data['mean_image'], mean='pixel')
out = deepdream(img_pre, 7, model, learning_rate=1000, num_iterations=300)
