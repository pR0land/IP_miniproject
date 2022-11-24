import numpy as np
import math
import bordering as b
import cv2 as cv
def differenceOfGaussian(image, kernelsize, SD):

    blurredPictures = [image]
    borderimage = b.addborder_reflect(image, kernelsize)

    for k in range(0, 4):
        guassiankernel = makeGuassianKernel(kernelsize, SD * 1.5**k)
        blurredPictures.append(convolve(borderimage, guassiankernel))

    DoG = []
    for i, picture in enumerate(blurredPictures):
        if i + 1 == len(blurredPictures):
            break
        DoG.append(cv.subtract(picture, blurredPictures[i + 1]))
    return DoG


def makeGuassianKernel(kernelsize, SD):
    radius = int((kernelsize - 1) / 2)  # kernel radius
    guassian = [1/(math.sqrt(2*math.pi) * SD) * math.exp(-0.5 * (x/SD)**2) for x in range(-radius, radius+1)]
    guassianKernel = np.zeros((kernelsize, kernelsize))

    for y in range(guassianKernel.shape[0]):
        for x in range(guassianKernel.shape[1]):
            guassianKernel[y, x] = guassian[y] * guassian[x]
    guassianKernel *= 1/guassianKernel[0, 0]
    return guassianKernel

def convolve(image, kernel):
    kernelSize = kernel.shape[0]
    sumKernel = kernel/np.sum(kernel)
    output = np.zeros((image.shape[0] - kernelSize + 1, image.shape[1] - kernelSize + 1, image.shape[2]), dtype=np.uint8)


    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            for channel in range(output.shape[2]):
                slice = image[y: y+kernelSize, x: x+kernelSize, channel]
                output[y, x, channel] = np.sum(slice * sumKernel)
    return output