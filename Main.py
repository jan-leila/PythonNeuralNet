from TrainingData import data, answers
import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))
def deriv(x):
    return x*(1-x)

class neuralNetwork():
    def __init__(self, layerSizes):
        #create all of the Synapse 
        print "Creating Synapse"
        self.Synapse = [None] * (len(layerSizes) - 1)
        for index in xrange(len(layerSizes) - 1):
            self.Synapse[index] = 2 * np.random.random((layerSizes[index],layerSizes[index + 1]))
        print "Finished Creating Synapse"
        print self.Synapse
        
        #create all of the Nodes 
        print "Creating Nodes"
        self.Nodes = [None] * len(layerSizes)
        for index in xrange(len(layerSizes)):
            self.Nodes[index] = np.empty(layerSizes[index])
        print "Finished Creating nodes"
        print self.Nodes
        
        #create all of the Errors 
        print "Creating Errors"
        self.Error = [None] * (len(self.Synapse))
        for index in xrange(len(self.Synapse)):
            self.Error[index] = np.zeros((layerSizes[index], layerSizes[index + 1]))
        print "Finished Creating Errors"
        print self.Error
        
        #create all of the Deltas 
        print "Creating Deltas"
        self.Delta = [None] * (len(self.Synapse))
        for index in xrange(len(self.Synapse)):
            self.Delta[index] = np.zeros((layerSizes[index],layerSizes[index + 1]))
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
                print "error after " + str(count) + " trainning loops: " + "NA"
            
            #Correct numbers
            self.correctError()
            
        print "training done"
        
    def runData(self, data):
        self.Nodes[0] = np.array(data)
        for index in xrange(len(self.Synapse)):
            self.Nodes[index + 1] = 1/(1+np.exp(-(np.dot(self.Nodes[index],self.Synapse[index]))))
    
    def calcError(self, answer):
        print "calculating error"
        for index in range(len(self.Delta) - 1, -1, -1):
            if index == len(self.Delta):
                self.Error[index] = answer - self.Nodes[index + 1]
                self.Delta[index] = self.Error[index] * deriv(self.Nodes[index + 1])
            else:
                self.Error[index - 1] = self.Delta[index].dot(self.Synapse[index].T)
                self.Delta[index - 1] = self.Error[index - 1] * deriv(self.Nodes[index])
    
    def correctError(self):
        print "correcting error"
        for index in xrange(len(self.Synapse)):
            print self.Nodes[index]
            print self.Delta[index]
            print self.Nodes[index].T.dot(self.Delta[index])
            self.Synapse[index] += self.Nodes[index].dot(self.Delta[index])

net = neuralNetwork([5,25,5,1])
net.train([[[1]*5],[[1]*5]],50,10)