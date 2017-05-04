__coauthor__ = 'Deeplayer'
# 6.22.2016 #

import time
from fc_net import *
from solver import *
import matplotlib.pyplot as plt
from data_utils import get_CIFAR10_data


# Load the (preprocessed) CIFAR10 data.
data = get_CIFAR10_data()

# Try training a very deep net with batchnorm
hidden_dims = [100, 100, 100, 100, 100]

weight_scale = 2e-2
bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, reg=0.001, use_batchnorm=True, dropout=0.3)
bn_solver = Solver(bn_model, data,
                num_epochs=40, batch_size=400,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=1000)

t1 = time.time()
bn_solver.train()
t2 = time.time()

best_model = bn_model

y_val_pred = np.argmax(best_model.loss(data['X_val'], y='best'), axis=1)
y_test_pred = np.argmax(best_model.loss(data['X_test'], y='best'), axis=1)
y_test_pred1 = np.argmax(best_model.loss(data['X_test']), axis=1)

print 'Validation set accuracy: ', (y_val_pred == data['y_val']).mean()
print 'Test set accuracy: ', (y_test_pred == data['y_test']).mean()
print 'Test set accuracy: ', (y_test_pred1 == data['y_test']).mean()


# Visualize the weights of the best network
from vis_utils import visualize_grid

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(3, 32, 32, -1).transpose(3, 1, 2, 0)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

show_net_weights(best_model)
