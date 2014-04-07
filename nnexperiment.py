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
    trainer.trainUntilConvergence(maxEpochs=3)
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
    trainer.trainUntilConvergence(maxEpochs=3)
    result = percentError(trainer.testOnClassData(dataset=test_ds),
                          test_ds['class'])
    #result = validate(trainer, train_ds, 5, 10)
    print 'One hidden layer - Percent Error', result
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
    
    perms = np.array_split(np.arange(len(data)), 10)
    simple_results = []
    one_hl_results = []
    for i in xrange(10):
        train_ds = ClassificationDataSet(64, 1, nb_classes = 10)
        test_ds = ClassificationDataSet(64, 1, nb_classes = 10)
        
        train_perms_idxs = range(10)
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
    
    for i in xrange(len(simple_results)):
        print 'Simple %d : Hidden %d' % (simple_results[i], one_hl_results[i])
        
        

if __name__ == '__main__':
    sys.exit(main())