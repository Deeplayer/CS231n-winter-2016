
__coauthor__ = 'Deeplayer'
# 6.25.2016 #

from layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

       conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, use_batchnorm=False,
                 weight_scale=1e-3, reg=0.0, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
                        of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # Initialize weights and biases
        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros((1, num_filters))
        self.params['W2'] = weight_scale * np.random.randn(num_filters*H*W/4, hidden_dim)
        self.params['b2'] = np.zeros((1, hidden_dim))
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros((1, num_classes))

        if self.use_batchnorm:
            self.params['gamma1'] = np.ones((1, num_filters))
            self.params['beta1'] = np.zeros((1, num_filters))
            self.params['gamma2'] = np.ones((1, hidden_dim))
            self.params['beta2'] = np.zeros((1, hidden_dim))

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'}, {'mode': 'train'}]

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        if self.use_batchnorm:
            gamma1, beta1 = self.params['gamma1'], self.params['beta1']
            gamma2, beta2 = self.params['gamma2'], self.params['beta2']
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # compute the forward pass
        if self.use_batchnorm:
            a1, cache1 = conv_forward_fast(X, W1, b1, conv_param)
            bn1, cache_bn1 = spatial_batchnorm_forward(a1, gamma1, beta1, self.bn_params[0])
            a2, cache2 = relu_forward(bn1)
            a3, cache3 = max_pool_forward_fast(a2, pool_param)
            a4, cache4 = affine_forward(a3, W2, b2)
            bn2, cache_bn2 = batchnorm_forward(a4, gamma2, beta2, self.bn_params[1])
            a5, cache5 = relu_forward(bn2)
            scores, cache6 = affine_forward(a5, W3, b3)
        else:
            a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
            a2, cache2 = affine_relu_forward(a1, W2, b2)
            scores, cache3 = affine_forward(a2, W3, b3)

        if y is None:
            return scores

        grads = {}
        # compute the backward pass
        data_loss, dscores = softmax_loss(scores, y)
        if self.use_batchnorm:
            da5, dW3, db3 = affine_backward(dscores, cache6)
            dbn2 = relu_backward(da5, cache5)
            da4, dgamma2, dbeta2 = batchnorm_backward(dbn2, cache_bn2)
            da3, dW2, db2 = affine_backward(da4, cache4)
            da2 = max_pool_backward_fast(da3, cache3)
            dbn1 = relu_backward(da2, cache2)
            da1, dgamma1, dbeta1 = spatial_batchnorm_backward(dbn1, cache_bn1)
            dX, dW1, db1 = conv_backward_fast(da1, cache1)
        else:
            da2, dW3, db3 = affine_backward(dscores, cache3)
            da1, dW2, db2 = affine_relu_backward(da2, cache2)
            dX, dW1, db1 = conv_relu_pool_backward(da1, cache1)

        # Add regularization
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        reg_loss = 0.5 * self.reg * sum(np.sum(W * W) for W in [W1, W2, W3])

        loss = data_loss + reg_loss

        grads['W1'], grads['b1'] = dW1, db1
        grads['W2'], grads['b2'] = dW2, db2
        grads['W3'], grads['b3'] = dW3, db3
        if self.use_batchnorm:
            grads['gamma1'], grads['beta1'] = dgamma1, dbeta1
            grads['gamma2'], grads['beta2'] = dgamma2, dbeta2

        return loss, grads
