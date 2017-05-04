import numpy as np

class ConvertCIFAR10:

    def __init__(self):
        pass

    def unpickle(file):
        import cPickle
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict

    # get the training data
    dataTrain = []
    labelTrain = []
    for i in range(1,6):
        dic = unpickle('E:\PycharmProjects\ML\CS231n\cifar-10-batches-py\data_batch_' + str(i))
        for item in dic['data']:
            dataTrain.append(item)
        for item in dic['labels']:
            labelTrain.append(item)

    # get the test data
    dataTest = []
    labelTest = []
    dic = unpickle('E:\PycharmProjects\ML\CS231n\cifar-10-batches-py\\test_batch')
    for item in dic['data']:
        dataTest.append(item)
    for item in dic['labels']:
        labelTest.append(item)

    #print 'labelTest: %d' % (len(labelTest))
    dataTr = np.asarray(dataTrain)      # N by D (50000,3072)
    dataTs = np.asarray(dataTest)       # N by D (10000,3072)
    labelTr = np.asarray(labelTrain)    # 1 by N (1,50000)
    labelTs = np.asarray(labelTest)     # 1 by N (1,10000)

