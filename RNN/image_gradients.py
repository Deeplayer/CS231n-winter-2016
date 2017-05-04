from data_utils import *
import time, os, json
import matplotlib.pyplot as plt
from pretrained_cnn import PretrainedCNN
from image_utils import blur_image, deprocess_image

data = load_tiny_imagenet('E:/PycharmProjects/ML/CS231n/RNN/datasets/tiny-imagenet-100-A', subtract_mean=True)
for i, names in enumerate(data['class_names']):
    print i, ' '.join('"%s"' % name for name in names)

# Visualize some examples of the training data
classes_to_show = 7
examples_per_class = 5
class_idxs = np.random.choice(len(data['class_names']), size=classes_to_show, replace=False)
for i, class_idx in enumerate(class_idxs):
  train_idxs, = np.nonzero(data['y_train'] == class_idx)
  train_idxs = np.random.choice(train_idxs, size=examples_per_class, replace=False)
  for j, train_idx in enumerate(train_idxs):
    img = deprocess_image(data['X_train'][train_idx], data['mean_image'])
    plt.subplot(examples_per_class, classes_to_show, 1 + i + classes_to_show * j)
    if j == 0:
      plt.title(data['class_names'][class_idx][0])
    plt.imshow(img)
    plt.gca().axis('off')

plt.show()

model = PretrainedCNN(h5_file='E:/PycharmProjects/ML/CS231n/RNN/datasets/pretrained_model.h5')

"""
Saliency Maps
"""
def compute_saliency_maps(X, y, model):
      """
      Compute a class saliency map using the model for images X and labels y.

      Input:
      - X: Input images, of shape (N, 3, H, W)
      - y: Labels for X, of shape (N,)
      - model: A PretrainedCNN that will be used to compute the saliency map.

      Returns:
      - saliency: An array of shape (N, H, W) giving the saliency maps for the input
        images.
      """
      N,C,H,W = X.shape
      saliency = np.zeros((N,H,W))

      # Compute the score by a single forward pass
      scores, cache = model.forward(X, mode='test') # Score size (N,100)

      # The loss function we want to optimize, lambda = 10
      loss = (scores[np.arange(N), y] - 10*np.sqrt(np.sum(X**2)))   # Size (N,)

      # The gradient of this loss wih respect to the input image simply writes
      dscores = np.zeros_like(scores)
      dscores[np.arange(N), y] = 1.0
      dX, grads = model.backward(dscores, cache)
      saliency += np.max(np.abs(dX), axis=1)
      print np.abs(dX).shape
      return saliency


def show_saliency_maps(mask):
    mask = np.asarray(mask)
    X = data['X_val'][mask]
    y = data['y_val'][mask]

    saliency = compute_saliency_maps(X, y, model)

    for i in xrange(mask.size):
        plt.subplot(2, mask.size, i + 1)
        plt.imshow(deprocess_image(X[i], data['mean_image']))
        plt.axis('off')
        plt.title(data['class_names'][y[i]][0])
        plt.subplot(2, mask.size, mask.size + i + 1)
        plt.title(mask[i])
        plt.imshow(saliency[i])
        plt.axis('off')
    plt.gcf().set_size_inches(10, 4)
    plt.show()

# Show some random images
mask = np.random.randint(data['X_val'].shape[0], size=5)
show_saliency_maps(mask)

# These are some cherry-picked images that should give good results
show_saliency_maps([128, 3225, 2417, 1640, 4619])
