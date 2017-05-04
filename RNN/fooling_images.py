import numpy as np
from data_utils import *
import matplotlib.pyplot as plt
from image_utils import blur_image, deprocess_image
from pretrained_cnn import PretrainedCNN

def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image, of shape (1, 3, 64, 64)
    - target_y: An integer in the range [0, 100)
    - model: A PretrainedCNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
      by the model.
    """
    X_fooling = X.copy()
    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. Use gradient ascent on the target class score, using   #
    # the model.forward method to compute scores and the model.backward method   #
    # to compute image gradients.                                                #
    #                                                                            #
    # HINT: For most examples, you should be able to generate a fooling image    #
    # in fewer than 100 iterations of gradient ascent.                           #
    ##############################################################################

    N,C,H,W = X_fooling.shape      # N=1
    i = 0
    y_pred = -1
    lr = 200.0
    while (y_pred != target_y) & (i<200):

        scores, cache = model.forward(X_fooling, mode='test') # Score size (N,100)

        # The loss function we want to optimize(maximize)
        # loss = scores[np.arange(N), target_y]                 # Size (N,)
        # print loss
        # The gradient of this loss wih respect to the input image
        dscores = np.zeros_like(scores)
        dscores[np.arange(N), target_y] = 1.0
        dX, grads = model.backward(dscores, cache)
        X_fooling += lr*dX
        y_pred = model.loss(X_fooling).argmax(axis=1)
        i+=1
        print 'Iteration %d: current class: %d; target class: %d ' % (i, y_pred, target_y)

    return X_fooling

model = PretrainedCNN(h5_file='E:/PycharmProjects/ML/CS231n/RNN/datasets/pretrained_model.h5')
data = load_tiny_imagenet('E:/PycharmProjects/ML/CS231n/RNN/datasets/tiny-imagenet-100-A', subtract_mean=True)

# Find a correctly classified validation image
while True:
  i = np.random.randint(data['X_val'].shape[0])
  X = data['X_val'][i:i+1]
  y = data['y_val'][i:i+1]
  y_pred = model.loss(X)[0].argmax()
  if y_pred == y: break

target_y = 67
X_fooling = make_fooling_image(X, target_y, model)

# Make sure that X_fooling is classified as y_target
scores = model.loss(X_fooling)
assert scores[0].argmax() == target_y, 'The network is not fooled!'

# Show original image, fooling image, and difference
plt.subplot(1, 3, 1)
plt.imshow(deprocess_image(X, data['mean_image']))
plt.axis('off')
plt.title(data['class_names'][y][0])
plt.subplot(1, 3, 2)
plt.imshow(deprocess_image(X_fooling, data['mean_image'], renorm=True))
plt.title(data['class_names'][target_y][0])
plt.axis('off')
plt.subplot(1, 3, 3)
plt.title('Difference')
plt.imshow(deprocess_image(X - X_fooling, data['mean_image']))
plt.axis('off')
plt.show()