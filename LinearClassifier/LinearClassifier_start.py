
__author__ = 'Deeplayer'
# 5.20.2016

import numpy as np
import matplotlib.pyplot as plt
import math
from Linear_Classifier import *
from data_utils import load_CIFAR10

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
plt.rcParams['figure.figsize'] = (10.0, 8.0)   # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Load the raw CIFAR-10 data.
cifar10_dir = 'E:/PycharmProjects/ML/CS231n/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
# As a sanity check, we print out the size of the training and test data.
print 'Training data shape: ', X_train.shape   # (50000,32,32,3)
print 'Training labels shape: ', y_train.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape
print '#----------------------------------------------------------------------------------#'

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

# Split the data into train, val, and test sets.
num_training = 49000
num_validation = 1000
num_test = 1000
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]                  # (1000,32,32,3)
y_val = y_train[mask]                  # (1,1000)
mask = range(num_training)
X_train = X_train[mask]                # (49000,32,32,3)
y_train = y_train[mask]                # (1,49000)
mask = range(num_test)
X_test = X_test[mask]                  # (1000,32,32,3)
y_test = y_test[mask]                  # (1,1000)

# Preprocessing1: reshape the image data into rows
dataTr = np.reshape(X_train, (X_train.shape[0], -1))   # (49000,3072)
dataVal = np.reshape(X_val, (X_val.shape[0], -1))      # (1000,3072)
dataTs = np.reshape(X_test, (X_test.shape[0], -1))     # (10000,3072)
labelTr = y_train                                      # (1,49000)
labelVal = y_val                                       # (1,1000)
labelTs = y_test                                       # (1,10000)

# Preprocessing2: subtract the mean image
mean_image = np.mean(dataTr, axis=0)       # (1,3072)
dataTr -= mean_image
dataVal -= mean_image
dataTs -= mean_image

# Visualize the mean image
plt.figure(figsize=(4, 4))
plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))
plt.title('Mean image')
plt.show()

# Bias trick, extending the data
one_vectorTr = np.ones((dataTr.shape[0], 1))
one_vectorVal = np.ones((dataVal.shape[0], 1))
one_vectorTs = np.ones((dataTs.shape[0], 1))
Tr_extended = np.concatenate((dataTr, one_vectorTr), axis=1)        # (50000,3073)
Tr_labels = labelTr                                                 # (1,49000)
Val_extended = np.concatenate((dataVal, one_vectorVal), axis=1).T   # (3073,1000)
Val_labels = labelVal                                               # (1,1000)
Ts_extended = np.concatenate((dataTs, one_vectorTs), axis=1).T      # (3073,10000)
Ts_labels = labelTs                                                 # (1,10000)

# Use the validation set to tune hyperparameters (regularization strength and learning rate).
learning_rates = [1e-7, 5e-5]
regularization_strengths = [5e4, 1e5]
results = {}
best_val = -1     # The highest validation accuracy that we have seen so far.
best_svm = None   # The LinearSVM object that achieved the highest validation rate.
iters = 1500
for lr in learning_rates:
    for rs in regularization_strengths:
        svm = LinearSVM()
        svm.train(Tr_extended, Tr_labels, learning_rate=lr, reg=rs, num_iters=iters)
        Tr_pred = svm.predict(Tr_extended.T)
        acc_train = np.mean(Tr_labels == Tr_pred)
        Val_pred = svm.predict(Val_extended)
        acc_val = np.mean(Val_labels == Val_pred)
        results[(lr, rs)] = (acc_train, acc_val)
        if best_val < acc_val:
            best_val = acc_val
            best_svm = svm

#print results
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy)
print 'Best validation accuracy achieved during validation: %f' % best_val


# Visualize the validation results
x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]

# plot training accuracy
marker_size = 100          # default size of markers is 20
colors = [results[x][0] for x in results]
plt.subplot(2, 1, 1)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 training accuracy')

# plot validation accuracy
colors = [results[x][1] for x in results]
plt.subplot(212)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 validation accuracy')
plt.show()

# Evaluate the best svm on test set
Ts_pred = best_svm.predict(Ts_extended)
test_accuracy = np.mean(Ts_labels == Ts_pred)
print 'LinearSVM on raw pixels of CIFAR-10 final test set accuracy: %f' % test_accuracy   # 36.61%

# Visualize the learned weights for each class
w = best_svm.W[:-1, :]   # strip out the bias
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in xrange(10):
    plt.subplot(2, 5, i + 1)
    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])

plt.show()


"""
svm = LinearSVM()
loss_history = svm.train(Tr_extended, Tr_labels, learning_rate=1e-7, reg=5e4, num_iters=1500)
label_predict = svm.predict(Val_extended)

print 'Accuracy by LinearSVM: %f' % (np.mean(label_predict == Val_labels))

# A useful debugging strategy is to plot the loss as a function of iteration number:
plt.plot(loss_history)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.title('Loss by LinearSVM')
plt.show()
"""