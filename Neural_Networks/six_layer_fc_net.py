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

weight_scale = 5e-2
bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, reg=0.014, use_batchnorm=True)
bn_solver = Solver(bn_model, data,
                num_epochs=100, batch_size=400,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=1000)

t1 = time.time()
bn_solver.train()
t2 = time.time()

best_model = bn_model

y_val_pred = np.argmax(best_model.loss(data['X_val'], y=7), axis=1)
y_test_pred = np.argmax(best_model.loss(data['X_test'], y=7), axis=1)
#y_val_pred1 = np.argmax(best_model.loss(data['X_val']), axis=1)
y_test_pred1 = np.argmax(best_model.loss(data['X_test']), axis=1)

print 'Validation set accuracy: ', (y_val_pred == data['y_val']).mean()
print 'Test set accuracy: ', (y_test_pred == data['y_test']).mean()
#print 'Validation set accuracy: ', (y_val_pred1 == data['y_val']).mean()
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


weight_scale = 2e-2
model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, reg=0.01, use_batchnorm=False)
solver = Solver(model, data,
                num_epochs=15, batch_size=400,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=1000)

solver.train()

best_model = model
y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)
print 'Validation set accuracy: ', (y_val_pred == data['y_val']).mean()
print 'Test set accuracy: ', (y_test_pred == data['y_test']).mean()


# Run the following to visualize the results from two networks
# trained above. You should find that using batch normalization
# helps the network to converge much faster.
plt.subplot(3, 1, 1)
plt.title('Training loss')
plt.xlabel('Iteration')

plt.subplot(3, 1, 2)
plt.title('Training accuracy')
plt.xlabel('Epoch')

plt.subplot(3, 1, 3)
plt.title('Validation accuracy')
plt.xlabel('Epoch')

plt.subplot(3, 1, 1)
plt.plot(solver.loss_history, 'o', label='baseline')
plt.plot(bn_solver.loss_history, 'o', label='batchnorm')

plt.subplot(3, 1, 2)
plt.plot(solver.train_acc_history, '-o', label='baseline')
plt.plot(bn_solver.train_acc_history, '-o', label='batchnorm')

plt.subplot(3, 1, 3)
plt.plot(solver.val_acc_history, '-o', label='baseline')
plt.plot(bn_solver.val_acc_history, '-o', label='batchnorm')

for i in [1, 2, 3]:
    plt.subplot(3, 1, i)
    if i == 1:
        plt.legend(loc='upper center', ncol=4)
    else:
        plt.legend(loc='lower center', ncol=4)
plt.gcf().set_size_inches(15, 15)
plt.show()
