#from TrainingData import data, answers
import numpy as np

# Sigmoid function
# Makes all numbers < 1 and > -1
def sigmoid(x):
	return 1/(1+np.exp(-x))

# Derivative function
# Gets the derivative of values
def deriv(x):
    return x*(1-x)

# The class you make to make your own net
class neuralNetwork():
    def __init__(self, layerSizes, seed = 1, printOnCreation = False):
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

        #set the seed for numpy so we always get the same results when we generate the array
        np.random.seed(seed)
        
        # create all of the Nodes
        # Create a 1d array with one slot for each layer
        self.Nodes = [None] * len(layerSizes)
        for index in xrange(len(layerSizes)):
            # Fill each slot in the 1d array with another 1d array corresponding to the pre defined layer sizes
            self.Nodes[index] = np.empty(layerSizes[index])
        
        # create all of the Errors 
        # all node layers have an error layer of equal size except for the first node layer
        self.Error = [None] * (len(self.Nodes) - 1)

        # create all of the Synapse 
        # Synapse is the connections between each node in 2 layers
        # Create a 1d array with one slot for each layer
        self.Synapse = [None] * (len(layerSizes) - 1)
        for index in xrange(len(layerSizes) - 1):
            # Fill each slot in the 1d array with another 2d array corresponding to the last node layer multiplied by this node layers size
            self.Synapse[index] = 2 * np.random.random((layerSizes[index],layerSizes[index + 1])) - 1
        
        # create all of the Deltas
        # Every Synapse layer has a Delta layer of equal size
        self.Delta = [None] * (len(self.Synapse))

        # For debugging print out all the things so you can look at them when the network starts
        if printOnCreation:
            print "Nodes:"
            print self.Nodes
            print "Synapse:"
            print self.Synapse
            print "Delta:"
            print self.Delta
            print "Delta:"
            print self.Delta
    
    # Function to automatically train the network
    def train(self,trainingData, traingingTime = 5, printCount = 1):
        print "starting training"
        # Var to store the number to print on based on the count of prints we want
        printTime = traingingTime/printCount

        # Run through data processing and then correct based on error -traingingTime- times
        for count in xrange(traingingTime):
            # Run net on data
            self.runData(trainingData[0])

            # Check data output from -runData- and get the error
            self.calcError(trainingData[1])
            
            # if it is time to print, print the error from the last run though
            if(count % printTime == 0):
                print str(count) + "/" + str(traingingTime) + " training loops. Error: " + str(np.mean(np.abs(self.Error[len(self.Error) - 1])) * 100) + "%"
            
            # Change the Synapse based on the error
            self.correctError()
        
        print "training done"
        print "Trained for " + str(traingingTime) + " loops."
        print "Ending error: " + str(np.mean(np.abs(self.Error[len(self.Error) - 1])) * 100) + "%"
        
    def runData(self, data):
        # Set the first node layer value to the data
        self.Nodes[0] = data

        for index in xrange(len(self.Synapse)):
            # Multiply the last error by Synapseto get this layer
            self.Nodes[index + 1] = sigmoid(np.dot(self.Nodes[index],self.Synapse[index]))
    
    def calcError(self, answer):
        # Start at the end of the network and move backwards calculating the error
        for index in range(len(self.Error) - 1, -1, -1):
            # If this is the last layer then there error is just your answer - the actual answer
            if index == len(self.Error) - 1:
                self.Error[index] = answer - self.Nodes[len(self.Nodes) - 1]
            # If this isn't the last layer then the error is next delta * next synapse
            else:
                self.Error[index] = self.Delta[index + 1].dot(self.Synapse[index + 1].T)
            # Delta is the error * the derivative of the next layer
            self.Delta[index] = self.Error[index] * deriv(self.Nodes[index + 1])
    
    def correctError(self):
        # Add the Delta, Node dot product to the Synapse to get a more accurate answer next time
        for index in xrange(len(self.Synapse)):
            self.Synapse[index] += self.Nodes[index].T.dot(self.Delta[index])
    
    def getNetData(self):
        # Run through each layer of Synapse and print it out
        for index in xrange(len(self.Synapse)):
            print "Layer - " + str(index)
            print self.Synapse[index]


#-------------------------------------------------------------------------------------------------------------

# Not important stuff
# FOR DEMO ONLY

# this is some sample data to use for testing the net
data = np.array([
            [0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
dataSize = len(data[0])

answers = np.array([
            [0],
			[1],
			[1],
			[0]])
answerSize = len(answers[0])

# creating the net with designated layer sizes (First and last layer must match the data and answers)
net = neuralNetwork([dataSize,5,answerSize])

# training the network with the data 100000 times and updating us 10 times through the presses
net.train([data,answers],100000,10)

#print out the values of the layer connections
net.getNetData()