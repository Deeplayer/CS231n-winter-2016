__coauthor__ = 'Deeplayer'
# 6.25.2016 #

from layer_utils import *


class MultiLayerConvNet(object):
    """
    [[conv - relu]x3 - pool]x4 - affine - relu - affine - softmax

    """
    def __init__(self, input_dim=(3, 32, 32), num_filters=(64, 128, 192, 256),
                 filter_size=3, hidden_dim=512, num_classes=10,
                 dropout=(0, 0, 0), reg=0.0, dtype=np.float32, seed=None):
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
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.dropout_param = []

        # Initialize weights and biases
        C, H, W = input_dim

        self.params['W1'] = np.sqrt(2.0/(filter_size**2*num_filters[0])) * np.random.randn(num_filters[0], C, filter_size, filter_size)
        self.params['b1'] = np.zeros((1, num_filters[0]))
        self.params['W2'] = np.sqrt(2.0/(filter_size**2*num_filters[0])) * np.random.randn(num_filters[0], num_filters[0], filter_size, filter_size)
        self.params['b2'] = np.zeros((1, num_filters[0]))
        self.params['W3'] = np.sqrt(2.0/(filter_size**2*num_filters[1])) * np.random.randn(num_filters[1], num_filters[0], filter_size, filter_size)
        self.params['b3'] = np.zeros((1, num_filters[1]))
        self.params['W6'] = np.sqrt(2.0/(filter_size**2*num_filters[2])) * np.random.randn(num_filters[2], num_filters[1], filter_size, filter_size)
        self.params['b6'] = np.zeros((1, num_filters[2]))
        self.params['W9'] = np.sqrt(2.0/(filter_size**2*num_filters[3])) * np.random.randn(num_filters[3], num_filters[2], filter_size, filter_size)
        self.params['b9'] = np.zeros((1, num_filters[3]))

        for i in xrange(1,4):
            self.params['W'+str(3*i+1)] = np.sqrt(2.0/(filter_size**2*num_filters[i])) * np.random.randn(num_filters[i], num_filters[i], filter_size, filter_size)
            self.params['b'+str(3*i+1)] = np.zeros((1, num_filters[i]))
            self.params['W'+str(3*i+2)] = np.sqrt(2.0/(filter_size**2*num_filters[i])) * np.random.randn(num_filters[i], num_filters[i], filter_size, filter_size)
            self.params['b'+str(3*i+2)] = np.zeros((1, num_filters[i]))

        self.params['W12'] = np.sqrt(2.0/num_filters[3]) * np.random.randn(num_filters[3], hidden_dim)
        self.params['b12'] = np.zeros((1, hidden_dim))
        self.params['W13'] = np.sqrt(2.0/hidden_dim) * np.random.randn(hidden_dim, num_classes)
        self.params['b13'] = np.zeros((1, num_classes))

        for j in xrange(3):
            self.params['gamma'+str(3*j+3)] = np.ones((1, num_filters[j+1]))
            self.params['beta'+str(3*j+3)] = np.zeros((1, num_filters[j+1]))
            self.params['gamma'+str(3*j+4)] = np.ones((1, num_filters[j+1]))
            self.params['beta'+str(3*j+4)] = np.zeros((1, num_filters[j+1]))
            self.params['gamma'+str(3*j+5)] = np.ones((1, num_filters[j+1]))
            self.params['beta'+str(3*j+5)] = np.zeros((1, num_filters[j+1]))

        self.params['gamma1'] = np.ones((1, num_filters[0]))
        self.params['beta1'] = np.zeros((1, num_filters[0]))
        self.params['gamma2'] = np.ones((1, num_filters[0]))
        self.params['beta2'] = np.zeros((1, num_filters[0]))
        self.params['gamma12'] = np.ones((1, hidden_dim))
        self.params['beta12'] = np.zeros((1, hidden_dim))

        self.bn_params = []
        self.bn_params = [{'mode': 'train'} for i in xrange(12)]

        self.dropout_param = [{'mode': 'train', 'p': 1-dropout[0]}, {'mode': 'train', 'p': 1-dropout[1]}, {'mode': 'train', 'p': 1-dropout[2]}]
        if seed is not None:
            self.dropout_param['seed'] = seed

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

        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        gamma3, beta3 = self.params['gamma3'], self.params['beta3']
        gamma4, beta4 = self.params['gamma4'], self.params['beta4']
        gamma5, beta5 = self.params['gamma5'], self.params['beta5']
        gamma6, beta6 = self.params['gamma6'], self.params['beta6']
        gamma7, beta7 = self.params['gamma7'], self.params['beta7']
        gamma8, beta8 = self.params['gamma8'], self.params['beta8']
        gamma9, beta9 = self.params['gamma9'], self.params['beta9']
        gamma10, beta10 = self.params['gamma10'], self.params['beta10']
        gamma11, beta11 = self.params['gamma11'], self.params['beta11']
        gamma12, beta12 = self.params['gamma12'], self.params['beta12']

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']
        W6, b6 = self.params['W6'], self.params['b6']
        W7, b7 = self.params['W7'], self.params['b7']
        W8, b8 = self.params['W8'], self.params['b8']
        W9, b9 = self.params['W9'], self.params['b9']
        W10, b10 = self.params['W10'], self.params['b10']
        W11, b11 = self.params['W11'], self.params['b11']
        W12, b12 = self.params['W12'], self.params['b12']
        W13, b13 = self.params['W13'], self.params['b13']

        for dp_param in self.dropout_param:
            dp_param['mode'] = mode

        for bn_param in self.bn_params:
            bn_param['mode'] = mode

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        pool_param_global = {'pool_height': 4, 'pool_width': 4, 'stride': 4}

        # compute the forward pass
        a1, cache1 = conv_forward_fast(X, W1, b1, conv_param)
        bn1, cache_bn1 = spatial_batchnorm_forward(a1, gamma1, beta1, self.bn_params[0])
        a2, cache2 = relu_forward(bn1)
        cd1, cache_cd1 = dropout_forward(a2, self.dropout_param[0])      # convdrop1  0.3
        a3, cache3 = conv_forward_fast(cd1, W2, b2, conv_param)
        bn2, cache_bn2 = spatial_batchnorm_forward(a3, gamma2, beta2, self.bn_params[1])
        a4, cache4 = relu_forward(bn2)
        af, cachef = max_pool_forward_fast(a4, pool_param)

        af1, cachef1 = conv_forward_fast(af, W3, b3, conv_param)
        bnf, cache_bnf = spatial_batchnorm_forward(af1, gamma3, beta3, self.bn_params[2])
        af2, cachef2 = relu_forward(bnf)
        cd2, cache_cd2 = dropout_forward(af2, self.dropout_param[1])     # convdrop2  0.4
        a5, cache5 = conv_forward_fast(cd2, W4, b4, conv_param)
        bn3, cache_bn3 = spatial_batchnorm_forward(a5, gamma4, beta4, self.bn_params[3])
        a6, cache6 = relu_forward(bn3)
        cdf, cache_cdf = dropout_forward(a6, self.dropout_param[1])      # convdrop3  0.4
        af3, cachef3 = conv_forward_fast(cdf, W5, b5, conv_param)
        bnf2, cache_bnf2 = spatial_batchnorm_forward(af3, gamma5, beta5, self.bn_params[4])
        af4, cachef4 = relu_forward(bnf2)
        a7, cache7 = max_pool_forward_fast(af4, pool_param)

        a8, cache8 = conv_forward_fast(a7, W6, b6, conv_param)
        bn4, cache_bn4 = spatial_batchnorm_forward(a8, gamma6, beta6, self.bn_params[5])
        a9, cache9 = relu_forward(bn4)
        cd3, cache_cd3 = dropout_forward(a9, self.dropout_param[1])     # convdrop4  0.4
        a10, cache10 = conv_forward_fast(cd3, W7, b7, conv_param)
        bn5, cache_bn5 = spatial_batchnorm_forward(a10, gamma7, beta7, self.bn_params[6])
        a11, cache11 = relu_forward(bn5)
        cd4, cache_cd4 = dropout_forward(a11, self.dropout_param[1])    # convdrop5  0.4
        a12, cache12 = conv_forward_fast(cd4, W8, b8, conv_param)
        bn6, cache_bn6 = spatial_batchnorm_forward(a12, gamma8, beta8, self.bn_params[7])
        a13, cache13 = relu_forward(bn6)
        a14, cache14 = max_pool_forward_fast(a13, pool_param)

        a15, cache15 = conv_forward_fast(a14, W9, b9, conv_param)
        bn7, cache_bn7 = spatial_batchnorm_forward(a15, gamma9, beta9, self.bn_params[8])
        a16, cache16 = relu_forward(bn7)
        cd5, cache_cd5 = dropout_forward(a16, self.dropout_param[1])    # convdrop6  0.4
        a17, cache17 = conv_forward_fast(cd5, W10, b10, conv_param)
        bn8, cache_bn8 = spatial_batchnorm_forward(a17, gamma10, beta10, self.bn_params[9])
        a18, cache18 = relu_forward(bn8)
        cd6, cache_cd6 = dropout_forward(a18, self.dropout_param[1])    # convdrop7  0.4
        a19, cache19 = conv_forward_fast(cd6, W11, b11, conv_param)
        bn9, cache_bn9 = spatial_batchnorm_forward(a19, gamma11, beta11, self.bn_params[10])
        a20, cache20 = relu_forward(bn9)
        a21, cache21 = max_pool_forward_fast(a20, pool_param_global)

        out1, cache_drop1 = dropout_forward(a21, self.dropout_param[2])    # dropout 0.5
        a22, cache22 = affine_forward(out1, W12, b12)
        bn10, cache_bn10 = batchnorm_forward(a22, gamma12, beta12, self.bn_params[11])
        a23, cache23 = relu_forward(bn10)
        out2, cache_drop2 = dropout_forward(a23, self.dropout_param[2])    # dropout 0.5
        scores, cache24 = affine_forward(out2, W13, b13)

        if y is None:
            return scores

        grads = {}
        # compute the backward pass
        data_loss, dscores = softmax_loss(scores, y)

        dout2, dW13, db13 = affine_backward(dscores, cache24)
        da23 = dropout_backward(dout2, cache_drop2)             # dropout
        dbn10 = relu_backward(da23, cache23)
        da22, dgamma12, dbeta12 = batchnorm_backward(dbn10, cache_bn10)
        dout1, dW12, db12 = affine_backward(da22, cache22)
        da21 = dropout_backward(dout1, cache_drop1)             # dropout

        da20 = max_pool_backward_fast(da21, cache21)
        dbn9 = relu_backward(da20, cache20)
        da19, dgamma11, dbeta11 = spatial_batchnorm_backward(dbn9, cache_bn9)
        dcd6, dW11, db11 = conv_backward_fast(da19, cache19)
        da18 = dropout_backward(dcd6, cache_cd6)                # dconvdrop7  0.4
        dbn8 = relu_backward(da18, cache18)
        da17, dgamma10, dbeta10 = spatial_batchnorm_backward(dbn8, cache_bn8)
        dcd5, dW10, db10 = conv_backward_fast(da17, cache17)
        da16 = dropout_backward(dcd5, cache_cd5)                # dconvdrop6  0.4
        dbn7 = relu_backward(da16, cache16)
        da15, dgamma9, dbeta9 = spatial_batchnorm_backward(dbn7, cache_bn7)
        da14, dW9, db9 = conv_backward_fast(da15, cache15)

        da13 = max_pool_backward_fast(da14, cache14)
        dbn6 = relu_backward(da13, cache13)
        da12, dgamma8, dbeta8 = spatial_batchnorm_backward(dbn6, cache_bn6)
        dcd4, dW8, db8 = conv_backward_fast(da12, cache12)
        da11 = dropout_backward(dcd4, cache_cd4)                # dconvdrop5  0.4
        dbn5 = relu_backward(da11, cache11)
        da10, dgamma7, dbeta7 = spatial_batchnorm_backward(dbn5, cache_bn5)
        dcd3, dW7, db7 = conv_backward_fast(da10, cache10)
        da9 = dropout_backward(dcd3, cache_cd3)                 # dconvdrop4  0.4
        dbn4 = relu_backward(da9, cache9)
        da8, dgamma6, dbeta6 = spatial_batchnorm_backward(dbn4, cache_bn4)
        da7, dW6, db6 = conv_backward_fast(da8, cache8)

        daf4 = max_pool_backward_fast(da7, cache7)
        dbnf2 = relu_backward(daf4, cachef4)
        daf3, dgamma5, dbeta5 = spatial_batchnorm_backward(dbnf2, cache_bnf2)
        dcdf, dW5, db5 = conv_backward_fast(daf3, cachef3)
        da6 = dropout_backward(dcdf, cache_cdf)                 # dconvdrop3  0.4
        dbn3 = relu_backward(da6, cache6)
        da5, dgamma4, dbeta4 = spatial_batchnorm_backward(dbn3, cache_bn3)
        dcd2, dW4, db4 = conv_backward_fast(da5, cache5)
        daf2 = dropout_backward(dcd2, cache_cd2)                 # dconvdrop2  0.4
        dbnf = relu_backward(daf2, cachef2)
        daf1, dgamma3, dbeta3 = spatial_batchnorm_backward(dbnf, cache_bnf)
        daf, dW3, db3 = conv_backward_fast(daf1, cachef1)

        da4 = max_pool_backward_fast(daf, cachef)
        dbn2 = relu_backward(da4, cache4)
        da3, dgamma2, dbeta2 = spatial_batchnorm_backward(dbn2, cache_bn2)
        dcd1, dW2, db2 = conv_backward_fast(da3, cache3)
        da2 = dropout_backward(dcd1, cache_cd1)                  # dconvdrop1  0.3
        dbn1 = relu_backward(da2, cache2)
        da1, dgamma1, dbeta1 = spatial_batchnorm_backward(dbn1, cache_bn1)
        dX, dW1, db1 = conv_backward_fast(da1, cache1)


        # Add regularization
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        dW4 += self.reg * W4
        dW5 += self.reg * W5
        dW6 += self.reg * W6
        dW7 += self.reg * W7
        dW8 += self.reg * W8
        dW9 += self.reg * W9
        dW10 += self.reg * W10
        dW11 += self.reg * W11
        dW12 += self.reg * W12
        dW13 += self.reg * W13

        reg_loss = 0.5 * self.reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11, W12, W13])

        loss = data_loss + reg_loss

        grads['W1'], grads['b1'] = dW1, db1
        grads['W2'], grads['b2'] = dW2, db2
        grads['W3'], grads['b3'] = dW3, db3
        grads['W4'], grads['b4'] = dW4, db4
        grads['W5'], grads['b5'] = dW5, db5
        grads['W6'], grads['b6'] = dW6, db6
        grads['W7'], grads['b7'] = dW7, db7
        grads['W8'], grads['b8'] = dW8, db8
        grads['W9'], grads['b9'] = dW9, db9
        grads['W10'], grads['b10'] = dW10, db10
        grads['W11'], grads['b11'] = dW11, db11
        grads['W12'], grads['b12'] = dW12, db12
        grads['W13'], grads['b13'] = dW13, db13


        grads['gamma1'], grads['beta1'] = dgamma1, dbeta1
        grads['gamma2'], grads['beta2'] = dgamma2, dbeta2
        grads['gamma3'], grads['beta3'] = dgamma3, dbeta3
        grads['gamma4'], grads['beta4'] = dgamma4, dbeta4
        grads['gamma5'], grads['beta5'] = dgamma5, dbeta5
        grads['gamma6'], grads['beta6'] = dgamma6, dbeta6
        grads['gamma7'], grads['beta7'] = dgamma7, dbeta7
        grads['gamma8'], grads['beta8'] = dgamma8, dbeta8
        grads['gamma9'], grads['beta9'] = dgamma9, dbeta9
        grads['gamma10'], grads['beta10'] = dgamma10, dbeta10
        grads['gamma11'], grads['beta11'] = dgamma11, dbeta11
        grads['gamma12'], grads['beta12'] = dgamma12, dbeta12

        return loss, grads
