__coauthor__ = 'Deeplayer'
# 6.22.2016 #

from fc_net import *
from solver import *
import matplotlib.pyplot as plt
from data_utils import get_CIFAR10_data

# Load the (preprocessed) CIFAR10 data.
data = get_CIFAR10_data()

# Try training a very deep net with batchnorm
hidden_dims = [100, 100, 100, 100, 100]

num_train = 5000
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

bn_solvers = {}
bn_model = FullyConnectedNet(hidden_dims, weight_scale=1e-3, reg=0.0, use_batchnorm=True)
bn_solver = Solver(bn_model, small_data,
                  num_epochs=10, batch_size=100,
                  update_rule='adam',
                  optim_config={
                    'learning_rate': 1e-3,
                  },
                  verbose=False, print_every=1000)
bn_solver.train()

for i in xrange(6):
    w = bn_model.grad['W'+str(i+1)].astype(np.float64)
    print 'layer' + str(i+1) + ':', np.mean(np.abs(w))
    print
