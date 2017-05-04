import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
from pretrained_cnn import PretrainedCNN
from data_utils import load_tiny_imagenet
from image_utils import blur_image, deprocess_image, preprocess_image

data = load_tiny_imagenet('E:/PycharmProjects/ML/CS231n/RNN/datasets/tiny-imagenet-100-A', subtract_mean=True)
model = PretrainedCNN(h5_file='E:/PycharmProjects/ML/CS231n/RNN/datasets/pretrained_model.h5')

def invert_features(target_feats, layer, model, **kwargs):
    """
    Perform feature inversion in the style of Mahendran and Vedaldi 2015, using
    L2 regularization and periodic blurring.

    Inputs:
    - target_feats: Image features of the target image, of shape (1, C, H, W);
    we will try to generate an image that matches these features
    - layer: The index of the layer from which the features were extracted
    - model: A PretrainedCNN that was used to extract features

    Keyword arguments:
    - learning_rate: The learning rate to use for gradient descent
    - num_iterations: The number of iterations to use for gradient descent
    - l2_reg: The strength of L2 regularization to use; this is lambda in the
    equation above.
    - blur_every: How often to blur the image as implicit regularization; set
    to 0 to disable blurring.
    - show_every: How often to show the generated image; set to 0 to disable
    showing intermediate reuslts.

    Returns:
    - X: Generated image of shape (1, 3, 64, 64) that matches the target features.
    """

    learning_rate = kwargs.pop('learning_rate', 10000)
    num_iterations = kwargs.pop('num_iterations', 500)
    l2_reg = kwargs.pop('l2_reg', 1e-7)
    blur_every = kwargs.pop('blur_every', 1)
    show_every = kwargs.pop('show_every', 50)

    X = np.random.randn(1, 3, 64, 64)
    for t in xrange(num_iterations):

        # Forward until target layer
        feats, cache = model.forward(X, end=layer, mode='test')

        # Compute the loss
        loss = np.sum((feats-target_feats)**2) + l2_reg*np.sum(X**2)

        # Compute the gradient of the loss with respect to the activation
        dfeats = 2*(feats-target_feats)
        dX, grads = model.backward(dfeats, cache)
        dX += 2*l2_reg*X

        X -= learning_rate*dX

        # As a regularizer, clip the image
        X = np.clip(X, -data['mean_image'], 255.0 - data['mean_image'])

        # As a regularizer, periodically blur the image
        if (blur_every > 0) and t % blur_every == 0:
            X = blur_image(X)

        if (show_every > 0) and (t % show_every == 0 or t + 1 == num_iterations):
            print loss
            plt.imshow(deprocess_image(X, data['mean_image']))
            plt.gcf().set_size_inches(3, 3)
            plt.axis('off')
            plt.title('Iteration: %d' % t)
            plt.show()

"""
filename = 'puppy.jpg'
layer = 3    # layers start from 0 so these are features after 4 convolutions
img = imresize(imread(filename), (64, 64))

plt.imshow(img)
plt.gcf().set_size_inches(3, 3)
plt.title('Original image')
plt.axis('off')
plt.show()

# Preprocess the image before passing it to the network:
# subtract the mean, add a dimension, etc
img_pre = preprocess_image(img, data['mean_image'])

# Extract features from the image
feats, _ = model.forward(img_pre, end=layer)

# Invert the features
kwargs = {
  'num_iterations': 400,
  'learning_rate': 5000,
  'l2_reg': 1e-8,
  'show_every': 100,
  'blur_every': 10,
}
X = invert_features(feats, layer, model, **kwargs)
"""

filename = 'puppy.jpg'
layer = 6 # layers start from 0 so these are features after 7 convolutions
img = imresize(imread(filename), (64, 64))

plt.imshow(img)
plt.gcf().set_size_inches(3, 3)
plt.title('Original image')
plt.axis('off')
plt.show()

# Preprocess the image before passing it to the network:
# subtract the mean, add a dimension, etc
img_pre = preprocess_image(img, data['mean_image'])

# Extract features from the image
feats, _ = model.forward(img_pre, end=layer)

# Invert the features
# You will need to play with these parameters.
kwargs = {
  'num_iterations': 1000,
  'learning_rate': 7500,
  'l2_reg': 1e-8,
  'show_every': 200,
  'blur_every': 1,
}
X = invert_features(feats, layer, model, **kwargs)
