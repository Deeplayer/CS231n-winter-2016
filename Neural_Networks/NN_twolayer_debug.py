__author__ = 'Deeplayer'
# 6.15.2016 #

import numpy as np
import matplotlib.pyplot as plt
from TwoLayerNN import TwoLayerNet
from data_utils import load_CIFAR10

# Load the data
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'E:/PycharmProjects/ML/CS231n/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]                   # (1000,32,32,3)
    y_val = y_train[mask]                   # (1000L,)
    mask = range(num_training)
    X_train = X_train[mask]                 # (49000,32,32,3)
    y_train = y_train[mask]                 # (49000L,)
    mask = range(num_test)
    X_test = X_test[mask]                   # (1000,32,32,3)
    y_test = y_test[mask]                   # (1000L,)

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)      # (49000,3072)
    X_val = X_val.reshape(num_validation, -1)        # (1000,3072)
    X_test = X_test.reshape(num_test, -1)            # (1000,3072)

    return X_train, y_train, X_val, y_val, X_test, y_test

# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape


# To train our network we will use SGD with momentum. In addition, we will
# adjust the learning rate with an exponential learning rate schedule as optimization proceeds;
# after each epoch, we will reduce the learning rate by multiplying it by a decay rate.
input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_epochs=4, batch_size=200, mu=0.5, mu_increase=1.0,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.5, verbose=True)

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print 'Validation accuracy: ', val_acc


# Debug the training.
"""
 With the default parameters we provided above, you should get a
 validation accuracy of about 0.29 on the validation set. This isn't very good.
 - One strategy for getting insight into what's wrong is to plot the loss function
   and the accuracies on the training and validation sets during optimization.
 - Another strategy is to visualize the weights that were learned in the first layer
   of the network. In most neural networks trained on visual data, the first layer
   weights typically show some visible structure when visualized.
"""
# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend(bbox_to_anchor=(1.0, 0.4))
plt.show()


# Visualize the weights of the network
from vis_utils import visualize_grid

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

show_net_weights(net)