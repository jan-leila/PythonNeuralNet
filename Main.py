#from TrainingData import data, answers
import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))

def deriv(x):
    return x*(1-x)

class neuralNetwork():
    def __init__(self, layerSizes, printOnCreation = False):
        # n - Node
        # N - Node Layer
        # E - Node Layer has error value
        # S - Synapse Layer
        # D - Synapse Layer has delta value
        #
        # N S N S N
        #   D E D E
        #   
        #     n
        #   /   \
        # n---n---n
        #   x   x
        # n---n---n
        #   x   x
        # n---n---n
        #   \    /
        #     n
        
        # create all of the Nodes 
        if printOnCreation:
            print "Creating Nodes"
        self.Nodes = [None] * len(layerSizes)
        for index in xrange(len(layerSizes)):
            self.Nodes[index] = np.empty(layerSizes[index])
        if printOnCreation:
            print "Finished Creating nodes"
            print self.Nodes
        
        # create all of the Errors 
        # all node layers have an error layer of equal size exept for the first node layer
        if printOnCreation:
            print "Creating Errors"
        self.Error = [None] * (len(self.Nodes) - 1)
        if printOnCreation:
            print "Finished Creating Errors"
            print self.Error

        # create all of the Synapse 
        # Synapse is the conections between each node in 2 layers
        if printOnCreation:
            print "Creating Synapse"
        self.Synapse = [None] * (len(layerSizes) - 1)
        for index in xrange(len(layerSizes) - 1):
            self.Synapse[index] = 2 * np.random.random((layerSizes[index],layerSizes[index + 1])) - 1
        if printOnCreation:
            print "Finished Creating Synapse"
            print self.Synapse
        
        # create all of the Deltas
        # Every Synapse layer has a Delta layer of equal size
        if printOnCreation:
            print "Creating Deltas"
        self.Delta = [None] * (len(self.Synapse))
        if printOnCreation:
            print "Finished Creating Deltas"
            print self.Delta
        
    def train(self,trainingData, traingingTime = 5, printCount = 1):
        print "starting training"
        #Var to store the number to print on based on the count of prints we want
        printTime = traingingTime/printCount
        #Run through data prossesing and then correct based on error traingingTime times
        for count in xrange(traingingTime):
            #Run net on data
            self.runData(trainingData[0])

            #Run error correction on answers
            self.calcError(trainingData[1])
            
            #if it is time to print print the error from the last run though
            if(count % printTime == 0):
                print "error after " + str(count) + " trainning loops: " + str(np.mean(np.abs(self.Error[len(self.Error) - 1])))
            
            #Correct numbers
            self.correctError()
            
        print "training done"
        print "Trainned for " + str(traingingTime) + " loops."
        print "Ending error: " + str(np.mean(np.abs(self.Error[len(self.Error) - 1])))
        
    def runData(self, data):
        self.Nodes[0] = data
        for index in xrange(len(self.Synapse)):
            self.Nodes[index + 1] = sigmoid(np.dot(self.Nodes[index],self.Synapse[index]))
    
    def calcError(self, answer):
        #print "calculating error"
        for index in range(len(self.Error) - 1, -1, -1):
            if index == len(self.Error) - 1:
                self.Error[index] = answer - self.Nodes[len(self.Nodes) - 1]
            else:
                self.Error[index] = self.Delta[index + 1].dot(self.Synapse[index + 1].T)
            self.Delta[index] = self.Error[index] * deriv(self.Nodes[index + 1])
    
    def correctError(self):
        #print "correcting error"
        for index in xrange(len(self.Synapse)):
            self.Synapse[index] += self.Nodes[index].T.dot(self.Delta[index])

np.random.seed(1)

net = neuralNetwork([3,6,10,7,5,1])

data = np.array([
            [0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
answers = np.array([
            [0],
			[1],
			[1],
			[0]])
net.train([data,answers],100000,10)