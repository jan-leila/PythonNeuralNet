from Network import Network

class AutoEncoder():
    def __init__(self, encoderSize, decoderSize, seed = 1, printOnCreation = False):
        self.encoder = Network(encoderSize, seed, printOnCreation)
        self.decoder = Network(decoderSize, seed, printOnCreation)

    def train(self, trainingData, trainingTime = 5, printCount = 1):
        def decodeError(originalData, encodedData):
            self.decoder.train([encodedData, originalData], 1, 0)
            return self.decoder.Error[0]
        self.encoder.lambdaTrain(decodeError, trainingData, trainingTime, printCount)

    def trainTillInterupt(self, trainingData, printTime = 10000):
        def decodeError(originalData, encodedData):
            self.decoder.train([encodedData, originalData], 1, 0)
            return self.decoder.Error[0]
        self.encoder.lambdaTrainTillInterupt(decodeError, trainingData, printTime)
    
    def encode(self, data):
        self.encoder.runData(data)
        return self.encoder.Nodes[-1]
    def decode(self, data):
        self.decoder.runData(data)
        return self.decoder.Nodes[-1]

autoE = AutoEncoder([10,5,3],[3,5,10],1,True)
