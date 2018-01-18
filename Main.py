from TrainingData import data, answers
import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))
def deriv(x):
    return x*(1-x)

class neuralNetwork():
    def __init__(self, layerSizes):
        #create all of the Synapse 
        self.Synapse = [None] * (len(layerSizes) - 1)
        for index in xrange(len(layerSizes) - 1):
            self.Synapse[index] = 2 * np.random.random((layerSizes[index],layerSizes[index + 1]))
        print "Finished Creating Synapse"
    def train(self,trainingData, traingingTime = 10000, printCount = 1000):
        for count in xrange(traingingTime):
            if(count % printCount == 0):
                print "error after " + str(count) + " trainning loops: " + "NA"
            self.runData(trainingData[0])
            self.fixError(trainingData[1])
        print "training"
    def runData(self, data):
        print "running data"
    def fixError(self, answer):
        print "calculating error"

net = neuralNetwork([5,25,5,1])
net.train([[[1]*5],[[1]*5]],50,10)