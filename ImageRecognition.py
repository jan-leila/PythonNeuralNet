import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def greyScaleImage(image):
    greyImage = [[0] * (len(image[0]) - 1)] * (len(image) - 1)
    for y in xrange(len(greyImage) - 1):
        for x in xrange(len(greyImage[y]) - 1):
            greyImage[y][x] = sum(image[y][x])
    return greyImage

def getImage(name):
    return plt.imread(name)

def flatenImage(image):
    return [item for sublist in image for item in sublist]

def getSpriteSheet(name,height,width,greyScale = False):
    spriteSheet = getImage(name)
    sprites = []

    coloms = len(spriteSheet[0])/height
    rows = len(spriteSheet)/width

    for y in xrange(coloms):
        for x in xrange(rows):
            sprite = spriteSheet[(y * height): ((y + 1)* height)][(x * width): ((x + 1)* width)].tolist()
            if sprite != []:
                sprites.append(sprite)
    print len(sprites)

    if greyScale:
        for index in xrange(len(sprites)):
            sprites[index] = greyScaleImage(sprites[index])

    for index in xrange(len(sprites)):
        sprites[index] = flatenImage(sprites[index])

    return sprites

#-------------------------------------------------------------------------------------------------------------

# Not important stuff
# FOR DEMO ONLY

# this is some sample data to use for testing the net
data = np.array(getSpriteSheet('HandWritingNumbers.png',32,32,True))

dataSize = len(data[0])

answers = np.array([
    [0,1,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0],
] * (dataSize / 10))

answerSize = len(answers[0])

from BasicNeuralNetwork import neuralNetwork

net = neuralNetwork([dataSize,25,25,5,answerSize]) 

net.train([data,answers],10000,1000)