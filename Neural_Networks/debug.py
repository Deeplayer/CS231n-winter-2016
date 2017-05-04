__coauthor__ = 'Deeplayer'
# 6.16.2016 #

import numpy as np
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


# Look for the best net
input_size = 32 * 32 * 3
hidden_size = 100
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

max_count = 100
for count in xrange(1, max_count + 1):
    reg = 10 ** np.random.uniform(-2, 0)       # (-4,0)
    lr = 10 ** np.random.uniform(-3.5, -3)       # (-5,-3)
    stats = net.train(X_train, y_train, X_val, y_val,
                      num_epochs=5, batch_size=200, mu=0.5, mu_increase=1.0,
                      learning_rate=lr, learning_rate_decay=0.95,
                      reg=reg, verbose=True)
    print 'val_acc: %f, lr: %s, reg: %s, (%d / %d)' % \
          (stats['val_acc_history'][-1], format(lr, 'e'), format(reg, 'e'), count, max_count)
