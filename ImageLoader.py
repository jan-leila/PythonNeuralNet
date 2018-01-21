import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# def getImageData(name,width,height):
#     data = [None]
#     img = plt.imread(name)
#     coloms = len(img)/height
#     rows = len(img[0])/width
#     img = img.ravel()
#     for y in range(0,rows):
#         for x in range(0,coloms):
#             sprite = img[(y * height * width + x * width) * 3: ((y + 1)* height * width + (x + 1) * width) * 3].tolist()
#             if data == []:
#                 data = [sprite]
#             else:
#                 data = np.vstack((data,sprite))
#     return data

def getImage(name):
    return plt.imread(name)

def greyScaleImage(image):
    greyImage = [[0] * (len(image[0]) - 1)] * (len(image) - 1)
    print greyImage
    for y in xrange(len(greyImage)):
        for x in xrange(len(greyImage[y])):
            greyImage[x][y] = sum(image[x][y])
            print greyImage[x][y]
    return greyImage

def getSpriteSheet(name,height,width,greyScale = False):
    spriteSheet = getImage(name)
    sprites = []

    coloms = len(spriteSheet)/height
    rows = len(spriteSheet[0])/width

    for y in xrange(coloms):
        for x in xrange(rows):
            sprites.append(spriteSheet[(y * height * width + x * width) * 3: ((y + 1)* height * width + (x + 1) * width) * 3].tolist())
    
    if greyScale:
        for index in xrange(len(sprites)):
            sprites[index] = greyScaleImage(sprites[index])
    
    #print sprites
    return sprites