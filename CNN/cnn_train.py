__coauthor__ = 'Deeplayer'
# 6.25.2016 #

import time
import numpy as np
from solver import Solver
from cnn3 import ThreeLayerConvNet
#from multiconv import MultiLayerConvNet
from multilayer_cnn import MultiLayerConvNet
import matplotlib.pyplot as plt
from data_utils import get_CIFAR10_data, get_CIFAR100_data
from vis_utils import visualize_grid


data = get_CIFAR10_data(data_aug='YES')     # YES 88.64 or NO 86.06
#data = get_CIFAR100_data(data_aug='YES')
model = MultiLayerConvNet(dropout=(0.3, 0.4, 0.5), reg=0.0015, num_classes=10)
solver = Solver(model, data,
                num_epochs=17, batch_size=128,
                update_rule='adam',
                optim_config={'learning_rate': 1e-3, },
                lr_decay=0.1,
                verbose=True, print_every=500)

train_start = time.time()
solver.train()
train_end = time.time()

print 'Training time: ', (train_end-train_start)/3600.0, 'hours'

y_val_pred = np.argmax(model.loss(data['X_val']), axis=1)
del data['X_train']
del data['X_val']
y_test_pred = np.zeros(len(data['y_test']))
for i in xrange(len(data['y_test'])/100):
    data_test = data['X_test'][i*100:100*(i+1), :, :, :]
    pred = np.argmax(model.loss(data_test), axis=1)
    y_test_pred[i*100:100*(i+1)] = pred

del data['X_test']
#print 'Validation set accuracy: ', (y_val_pred == data['y_val']).mean()
print 'Test set accuracy: ', (y_test_pred == data['y_test']).mean()
print 'Validation set accuracy: ', (y_val_pred == data['y_val']).mean()

plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.plot(solver.loss_history, 'o')
plt.xlabel('Iteration')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.plot([0.9] * len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()


#(model.params['W1'].transpose(0, 2, 3, 1))
grid = visualize_grid(model.params['W1'].transpose(0, 2, 3, 1))
plt.imshow(grid.astype('uint8'))
plt.axis('off')
plt.gcf().set_size_inches(5, 5)
plt.show()


'''
img = model.params['W1'].transpose(0, 2, 3, 1)[5]
low, high = np.min(img), np.max(img)
img = 255 * (img - low) / (high - low)
plt.imshow(img.astype('uint8'), interpolation='nearest')
plt.axis('off')
plt.gcf().set_size_inches(5, 5)
plt.show()
'''