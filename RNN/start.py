import time, os, json
import numpy as np
import matplotlib.pyplot as plt

from gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from rnn_layers import *
from captioning_solver import CaptioningSolver
from rnn import CaptioningRNN
from coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from image_utils import image_from_url

# Load COCO data from disk; this returns a dictionary
# We'll work with dimensionality-reduced features for this notebook, but feel
# free to experiment with the original features by changing the flag below.
data = load_coco_data(pca_features=True)
print min(data['train_image_idxs'])

# Print out all the keys and values from the data dictionary
for k, v in data.iteritems():
    if type(v) == np.ndarray:
        print k, type(v), v.shape, v.dtype
    else:
        print k, type(v), len(v), v


minibatch = sample_coco_minibatch(data, split='val')
gt_captions, features, urls = minibatch
caption_str = decode_captions(gt_captions, data['idx_to_word'])
print type(gt_captions)
print type(gt_captions[0])
print caption_str[1]

# Sample a minibatch and show the images and captions
batch_size = 1

captions, features, urls = sample_coco_minibatch(data, batch_size=batch_size)
for i, (caption, url) in enumerate(zip(captions, urls)):
    plt.imshow(image_from_url(url))
    plt.axis('off')
    caption_str = decode_captions(caption, data['idx_to_word'])
    plt.title(caption_str)
    plt.show()


