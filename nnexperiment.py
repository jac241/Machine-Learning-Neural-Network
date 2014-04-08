'''
Created on Apr 6, 2014

@author: jac241
'''
import sys
import copy
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SoftmaxLayer, FeedForwardNetwork, LinearLayer
from pybrain.structure import FullConnection, SigmoidLayer
from pybrain.tools.validation import CrossValidator
import random
from math import ceil
from numpy.random import permutation
import numpy as np


DATAFILE = "digits.data"


def paired_t_test(a, b):
    k = len(a)
    diff = 0
    for i in xrange(k):
        diff += a[i] - b[i]
    avg_diff = diff / k
    samp_var = 0
    for i in xrange(k):
        samp_var += (a[i] - b[i] - avg_diff)**2
    samp_var /= k-1
    std_dev = samp_var**.5
    t = avg_diff / (std_dev / k**0.5)
    print "t statistic = %f" % t
    if abs(t) > 2.776:                               # t statistic for 4 dof two tailed
        print "Difference is significant"
    else:
        print "Difference is not significant"


def validate(trainer, dataset, n_folds, max_epochs):
    l = dataset.getLength()
    inp = dataset.getField("input")
    tar = dataset.getField("target")
    indim = dataset.indim
    outdim = dataset.outdim
    assert l > n_folds
    perms = np.array_split(np.arange(l), n_folds)
    
    perf = 0.0
    for i in range(n_folds):
        # determine train indices
        train_perms_idxs = range(n_folds)
        train_perms_idxs.pop(i)
        temp_list = []
        for train_perms_idx in train_perms_idxs:
            temp_list.append(perms[ train_perms_idx ])
        train_idxs = np.concatenate(temp_list)

        # determine test indices
        test_idxs = perms[i]

        # train
        #print("training iteration", i)
        train_ds = ClassificationDataSet(indim, outdim)
        train_ds.setField("input"  , inp[train_idxs])
        train_ds.setField("target" , tar[train_idxs])
        train_ds._convertToOneOfMany()
        trainer = copy.deepcopy(trainer)
        trainer.setData(train_ds)
        if not max_epochs:
            trainer.train()
        else:
            trainer.trainEpochs(max_epochs)

        # test
        #print("testing iteration", i)
        test_ds = ClassificationDataSet(indim, outdim)
        test_ds.setField("input"  , inp[test_idxs])
        test_ds.setField("target" , tar[test_idxs])
        test_ds._convertToOneOfMany()
#            perf += self.getPerformance( trainer.module, dataset )
#         perf += self._calculatePerformance(trainer.module, dataset)
        perf += percentError(trainer.testOnClassData(dataset=test_ds),
                             test_ds['class'])

    perf /= n_folds
    return perf
    


def chunk_data(lines):
    chunks = []
    for i in xrange(10):
        start = int(ceil(float(len(lines)) / 10)) * i
        end = start + len(lines) / 10   # if (start + len(lines) / 10) < len(lines) else len(lines)
        chunks.append(lines[start:end])
    return chunks


def combine_chunks(chunks, start, end):
    data = []
    i = start
    while i != end:
        for line in chunks[i]:
            data.append(line)
        i = (i + 1) % 10
    return data 


def max_index(l):
    index = 0
    for i in len(l):
        if l[i] < l[index]:
            index = i
    return index            


def score(n, data, ds):
    ncorrect = 0
    for data in ds:
        input = data[0]
        output = data[1]
        pred_entry = n.activate(input)
        print 'Actual:', output, 'Predicted', pred_entry
        if pred_entry == output:
            ncorrect += 1
    print '%d / %d' % (ncorrect, len(ds))
    return score


def simple_network(data, digit, train_ds, test_ds):
#     n = buildNetwork(train_ds.indim, 1, train_ds.outdim, outclass=SoftmaxLayer)
    n = FeedForwardNetwork()
    inLayer = LinearLayer(64)
    outLayer = SoftmaxLayer(10)
    n.addInputModule(inLayer)
    n.addOutputModule(outLayer)
    n.addConnection(FullConnection(inLayer, outLayer))
    n.sortModules()
    trainer = BackpropTrainer(n, dataset=train_ds, momentum=0.1, verbose=True,
                              weightdecay=0.01)
    trainer.trainUntilConvergence(maxEpochs=25)
    result = percentError(trainer.testOnClassData(dataset=test_ds),
                          test_ds['class'])
#     result = validate(trainer, train_ds, 5, 10)
    print 'Simple network - Percent Error', result
    return result
    

def one_hidden_layer(data, digit, train_ds, test_ds):
    n = buildNetwork( train_ds.indim, 12, train_ds.outdim, outclass=SoftmaxLayer )
#     n = FeedForwardNetwork()
#     inLayer = LinearLayer(64)
#     outLayer = SoftmaxLayer(10)
#     hLayer = SigmoidLayer(12)
#     n.addInputModule(inLayer)
#     n.addModule(hLayer)
#     n.addOutputModule(outLayer)
#     n.addConnection(FullConnection(inLayer, hLayer))
#     n.addConnection(FullConnection(hLayer, outLayer))
#     n.sortModules()
    trainer = BackpropTrainer(n, dataset=train_ds, momentum=0.1,
                              weightdecay=0.01)
    trainer.trainUntilConvergence(maxEpochs=25)
    result = percentError(trainer.testOnClassData(dataset=test_ds),
                          test_ds['class'])
    #result = validate(trainer, train_ds, 5, 10)
    print 'One hidden layer - Percent Error', result
    #score(n, data, train_ds)
    return result

def creative_network(data, digit, train_ds, test_ds):
    # Create input vectors
    n = FeedForwardNetwork()
    inLayer = []
    for i in xrange(64):
        l = LinearLayer(1)
        inLayer.append(l)
        n.addInputModule(l)
    
    outLayer = SoftmaxLayer(10)
    i = 0
    j = 0
    middleLayers = [SigmoidLayer(1) for x in xrange(16)]
    
    for layer in middleLayers:
        n.addModule(layer)
    n.addOutputModule(outLayer)
    
    while i < 55:
        n.addConnection(FullConnection(inLayer[i], middleLayers[j]))
        n.addConnection(FullConnection(inLayer[i+1], middleLayers[j]))
        n.addConnection(FullConnection(inLayer[i+8], middleLayers[j]))
        n.addConnection(FullConnection(inLayer[i+9], middleLayers[j]))
        j += 1
        i += 2
        if (i % 16) >= 8:
            i += 8
    
    for layer in middleLayers:
        n.addConnection(FullConnection(layer, outLayer))
        
    n.sortModules()
    
    trainer = BackpropTrainer(n, dataset=train_ds, momentum=0.1,
                              weightdecay=0.01)
    trainer.trainUntilConvergence(maxEpochs=25)
    result = percentError(trainer.testOnClassData(dataset=test_ds),
                          test_ds['class'])
    #result = validate(trainer, train_ds, 5, 10)
    print 'Creative Network - Percent Error', result
    #score(n, data, train_ds)
    return result
    
        

def read_data(file):
    with open(file) as f:
        lines = f.readlines()
        random.shuffle(lines)    
    data = []
    digit = []
    for line in lines:
        v = line.split(',')
        data.append([int(x) for x in v[0:64]])
        digit.append(int(v[64]))
    
    return data, digit

#Rearranges data to put 2x2 squares next to each other linearly
# i.e. [i,i+1,i+8,i+9,i+2,i+3,i+10,i+11]... 
def rearrange_data(data):
    new = []
    i = 0
    while i < 55:
        new.append(data[i])
        new.append(data[i+1])
        new.append(data[i+8])
        new.append(data[i+9])
    
    
def main():
    random.seed(50)
    data, digit = read_data(DATAFILE)
    
#     ds = ClassificationDataSet(64, 1, nb_classes=10)
#     
#     
#     for i in xrange(len(data)):
#         ds.addSample(data[i], [digit[i]])
#     ds._convertToOneOfMany()
#     
#     simple_network(data, digit, ds)
#     one_hidden_layer(data, digit, ds)
    n_folds = 5
    perms = np.array_split(np.arange(len(data)), n_folds)
    simple_results = []
    one_hl_results = []
    creative_results = []
    for i in xrange(n_folds):
        train_ds = ClassificationDataSet(64, 1, nb_classes = 10)
        test_ds = ClassificationDataSet(64, 1, nb_classes = 10)
        
        train_perms_idxs = range(n_folds)
        train_perms_idxs.pop(i)
        temp_list = []
        for train_perms_idx in train_perms_idxs:
            temp_list.append(perms[ train_perms_idx ])
        train_idxs = np.concatenate(temp_list)
        
        for idx in train_idxs:
            train_ds.addSample(data[idx], [digit[idx]])
        train_ds._convertToOneOfMany()

        # determine test indices
        test_idxs = perms[i]
        for idx in test_idxs:
            test_ds.addSample(data[idx], [digit[idx]])
        test_ds._convertToOneOfMany()
        
        simple_results.append(simple_network(data, digit, train_ds, test_ds))
        one_hl_results.append(one_hidden_layer(data, digit, train_ds, test_ds))
        creative_results.append(creative_network(data, digit, train_ds, test_ds))
        
    for i in xrange(len(simple_results)):
        print 'Simple %f : Hidden %f : Creative %f' % (simple_results[i],
                                                    one_hl_results[i],
                                                    creative_results[i])
    print 'Simple mean: %f' % np.mean(simple_results)
    print 'One hidden layer mean: %f' % np.mean(one_hl_results)
    print 'Creative mean : %f' % np.mean(creative_results)
    
     
    print "Simple vs onehl"
    paired_t_test(simple_results, one_hl_results)
    print "simple vs creative"
    paired_t_test(simple_results, creative_results)
    print "onehl vs creative"
    paired_t_test(one_hl_results, creative_results)
        
        

if __name__ == '__main__':
    sys.exit(main())