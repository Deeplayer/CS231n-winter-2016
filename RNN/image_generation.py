import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
from pretrained_cnn import PretrainedCNN
from data_utils import load_tiny_imagenet
from image_utils import blur_image, deprocess_image, preprocess_image

data = load_tiny_imagenet('E:/PycharmProjects/ML/CS231n/RNN/datasets/tiny-imagenet-100-A', subtract_mean=True)
model = PretrainedCNN(h5_file='E:/PycharmProjects/ML/CS231n/RNN/datasets/pretrained_model.h5')

def create_class_visualization(target_y, model, **kwargs):
    """
    Perform optimization over the image to generate class visualizations.

    Inputs:
    - target_y: Integer in the range [0, 100) giving the target class
    - model: A PretrainedCNN that will be used for generation

    Keyword arguments:
    - learning_rate: Floating point number giving the learning rate
    - blur_every: An integer; how often to blur the image as a regularizer
    - l2_reg: Floating point number giving L2 regularization strength on the image;
    this is lambda in the equation above.
    - max_jitter: How much random jitter to add to the image as regularization
    - num_iterations: How many iterations to run for
    - show_every: How often to show the image
    """

    learning_rate = kwargs.pop('learning_rate', 10000)
    blur_every = kwargs.pop('blur_every', 1)
    l2_reg = kwargs.pop('l2_reg', 1e-6)
    max_jitter = kwargs.pop('max_jitter', 4)
    num_iterations = kwargs.pop('num_iterations', 200)
    show_every = kwargs.pop('show_every', 25)

    X = np.random.randn(1, 3, 64, 64)
    for t in xrange(num_iterations):
        # As a regularizer, add random jitter to the image
        ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
        X = np.roll(np.roll(X, ox, -1), oy, -2)

        # Compute the score
        scores, cache = model.forward(X, mode='test')
        loss = scores[0, target_y] - l2_reg*np.sum(X**2)
        dscores = np.zeros_like(scores)
        dscores[0, target_y] = 1.0
        dX, grads = model.backward(dscores, cache)
        dX -= 2*l2_reg*X

        X += learning_rate*dX

        # Undo the jitter
        X = np.roll(np.roll(X, -ox, -1), -oy, -2)

        # As a regularizer, clip the image
        X = np.clip(X, -data['mean_image'], 255.0 - data['mean_image'])

        # As a regularizer, periodically blur the image
        if t % blur_every == 0:
            X = blur_image(X)

        # Periodically show the image
        if t % show_every == 0:
            print 'The loss is %f' % loss
            plt.imshow(deprocess_image(X, data['mean_image']))
            plt.gcf().set_size_inches(3, 3)
            plt.axis('off')
            plt.title('Iteration: %d' % t)
            plt.show()

    return X

target_y = 43   # Tarantula
print data['class_names'][target_y]
X = create_class_visualization(target_y, model, show_every=25)