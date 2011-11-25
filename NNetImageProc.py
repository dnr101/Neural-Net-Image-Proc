#!/usr/bin/python

'''
Example of Neural Net from AI App Prog
CSDP 638
Created on Jul 14, 2010
@author: David Raizen
'''

import random
import math
from PIL import Image

#Constants
INPUT_NEURS = 3
HIDDEN_NEURS = 3
OUTPUT_NEURS = 1
MAX_SAMPLES = 200
#ACTION_LABLES = ['Attack', 'Run', 'Wander', 'Hide']
LEARN_RATE = 0.2 #Rho

#Neuron weight lists...
wih = [[0] * (HIDDEN_NEURS) for i in range(INPUT_NEURS+1)]
who = [[0] * (OUTPUT_NEURS) for i in range(HIDDEN_NEURS+1)]

#Neuron activations...
inputs = [0] * INPUT_NEURS
hidden = [0] * HIDDEN_NEURS
target = [0] * OUTPUT_NEURS
actual = [0] * OUTPUT_NEURS

#Unit Errors
erro = [0] * OUTPUT_NEURS
errh = [0] * HIDDEN_NEURS

#Some simple support functions...
def randWeight():
    return random.uniform(-0.5, 0.5)
    
def getSRand():
    return random.random()
    
def getRand(x):
    return int(x * getSRand())
    
def sqr(x):
    return x*x
    
def assignRandomWeights():
    """
    randomizes the weights on all neuron connections
    """
    for inp in xrange(INPUT_NEURS+1):
        for hid in xrange(HIDDEN_NEURS):
            wih[inp][hid] = randWeight()
            #wih[inp][hid] = 42
    for hid in xrange(HIDDEN_NEURS):
        for out in xrange(OUTPUT_NEURS):
            who[hid][out] = randWeight()

def sigmoid(val):
    """
    Squashing function for feed forward phase
    """
    return (1.0 / (1.0 + math.exp(-val)))

def sigmoidDerivative(val):
    """
    used in error backprop
    """
    return (val * (1.0 - val))

def feedForward():
    """
    Calculates the current status of the NNet, using the training data in samples[]
    """
    #Calculate input to hidden layer
    for hid in xrange(HIDDEN_NEURS):
        sum = 0.0
        for inp in xrange(INPUT_NEURS):
            sum += inputs[inp] * wih[inp][hid]
            #Add bias...
        sum += wih[INPUT_NEURS][hid]
        hidden[hid] = sigmoid(sum)
    
    #Calculate hidden to output layer
    for out in xrange(OUTPUT_NEURS):
        sum = 0.0
        for hid in xrange(HIDDEN_NEURS):
            sum += hidden[hid] * who[hid][out]
            
    #More bias...
        sum += who[HIDDEN_NEURS][out]
        actual[out] = sigmoid(sum)
    
def backPropagate():
    """
    Trains the NNet by passing error data backwards and adjusting the weights accordingly
    """
    #Calc the error on the output layer...
    for out in xrange(OUTPUT_NEURS):
        erro[out] = (target[out] - actual[out]) * sigmoidDerivative(actual[out])
    
    #Calc the error on the hidden layer...
    for hid in xrange(HIDDEN_NEURS):
        errh[hid] = 0.0
        for out in xrange(OUTPUT_NEURS):
            errh[hid] += erro[out] * who[hid][out]
        errh[hid] *= sigmoidDerivative(hidden[hid])
    
    #Update weights on the output layer...
    for out in xrange(OUTPUT_NEURS):
        for hid in xrange(HIDDEN_NEURS):
            who[hid][out] += (LEARN_RATE * erro[out] * hidden[hid])    
        #Update the bias...
        who[HIDDEN_NEURS][out] += (LEARN_RATE * erro[out])
    
    #Update weights on the hidden layer...
    for hid in xrange(HIDDEN_NEURS):
        for inp in xrange(INPUT_NEURS):
            wih[inp][hid] += (LEARN_RATE * errh[hid] * inputs[inp])
        #Update the bias...
        wih[INPUT_NEURS][hid] += (LEARN_RATE * errh[hid])
        
class element ():
    """
    Simple training data object
    """
    def __init__(self, pixel = (0, 0, 0), out = 0):
        self.pixel = pixel
        self.out = out
imgIn = Image.open('test.jpg')
pix = imgIn.load()
samples = []        
#Training data for the feedForward phase
#samples = [element([1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0] ), \
#           element([0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1] ), \
#           element([1, 1, 0, 1, 1, 0, 1], [0, 0, 1, 0] ), \
#           element([1, 1, 1, 1, 0, 0, 1], [0, 0, 1, 1] ), \
#           element([0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 0] ), \
#           element([1, 0, 1, 1, 0, 1, 1], [0, 1, 0, 1] ), \
#           element([1, 0, 1, 1, 1, 1, 1], [0, 1, 1, 0] ), \
#           element([1, 1, 1, 0, 0, 0, 0], [0, 1, 1, 1] ), \
#           element([1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 1] ), \
#           element([1, 1, 1, 0, 0, 1, 1], [1, 0, 0, 1] )]
for i in xrange(10, 20):
    for j in xrange(10, 20):
        testPx = pix[i, j]
        samples.append(element(testPx, 1))
for i in xrange(225, 235):
    for j in xrange(150, 160):
        testPx = pix[i, j]
        samples.append(element(testPx, 0))

def action(vect):
    """
    Binary display output function
    """
    
    if vect >= 0.5:
        vect = 1.0
    else:
        vect = 0.0
    
    return vect


def main():
    """
    Train the NNet and then allow the user to test it interactively 
    """
 
    sample = -1
    iterations = 0
    sum = 0
    fout = open('stats.txt', 'w')
    random.seed()
    assignRandomWeights()
    #Train the network...
    while True:
        sample += 1
        if sample >= MAX_SAMPLES:
            sample = 0

        for i in xrange(3):
            inputs[i] = samples[sample].pixel[i]
            
        target[0] = samples[sample].out
        
        feedForward()
        
        err = 0.0
        
        err += sqr(samples[sample].out - actual[0])
        err *= 0.5
        
        fout.write('%2.8f \n' %err)
        print 'mse = %2.8f \n' % err
        
        if iterations > 100000:
            break
        iterations += 1
        backPropagate()
    #Test the NNet against the training data    
    for i in xrange(MAX_SAMPLES):
        for j in xrange(3):
            inputs[j] = samples[sample].pixel[j]

        
        target[0] = samples[sample].out
        
        feedForward()
        
        if action(actual) != action(target):
            for j in xrange(3):
                print inputs[j], 
            print action(actual), action(target)
        else:
            sum += 1
        
    print 'Network is %2.2f%% correct\n' %((float(sum)/MAX_SAMPLES) * 100)
    
    #Allow the user to test the NNet interactively
    for x in xrange(640):
        for y in xrange(500):
            testPx = pix[x, y]
            for i in xrange(3):
                inputs[i] = testPx[i]
            feedForward()
            if action(actual[0]) < 0.5:
                pix[x, y] = (0, 0, 0)
    imgIn.save('test2.jpg')
            
    fout.close()
    
if __name__ == '__main__':
    main()
    