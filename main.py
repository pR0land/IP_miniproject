from collections import deque
import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy import stats
from statistics import mode
import Gaussian as g
import bordering as b
import Thresholding as t
import morphology as m
import time

gameboardList = []
gameareasList = []

tileTypes = ['water_0', 'water_1',
             'desert_0', 'desert_1',
             'forest_0', 'forest_1',
             'grass_0', 'grass_1', 'grass_2',
             'mine_0', 'mine_1', 'mine_2', 'mine_3',
             'waste_0', 'waste_1', 'waste_2']

binsize = 8


def addBoards():
    tempList = []
    for i in range(1, 75):
        picPath = 'King Domino dataset/Cropped and perspective corrected boards/' + str(i) + '.jpg'
        picTemp = cv.imread(picPath)
        tempList.append(picTemp)
    return tempList


def addAreas():
    tempList = []
    for i in range(1, 19):
        picPath = 'King Domino dataset/Full game areas/DSC_' + str(1262 + i) + '.JPG'
        picTemp = cv.imread(picPath)
        tempList.append(picTemp)
    return tempList


gameboardList.extend(addBoards())
gameareasList.extend(addAreas())


def showListOfImages(list):
    for i in range(len(list)):
        cv.imshow(str(i + 1), list[i])
    cv.waitKey(0)


def showSingleImage(list, index):
    if len(list) > index > 0:
        cv.imshow(str(index + 1), list[index])
        cv.waitKey(0)
    else:
        if len(list) < index:
            index = len(list) - 1
        elif index < 0:
            index = 0

        cv.imshow(str(index + 1), list[index])
        cv.waitKey(0)


def returnSingleImage(list, index):
    if len(list) < index or index < 0:
        if len(list) < index:
            index = len(list) - 1
        elif index < 0:
            index = 0
    return list[index]


def sliceROI(roi):
    output = []

    for y in range(0, roi.shape[0], int(roi.shape[0] / 5)):
        y_line = []
        for x in range(0, roi.shape[1], int(roi.shape[1] / 5)):
            slice = roi[y: y + int(roi.shape[0] / 5), x:x + int(roi.shape[1] / 5)].copy()
            y_line.append(slice)
        output.append(y_line)

    return output


def defineCenterAndBorder(slice):
    center = slice[int(slice.shape[0] / 10):int((slice.shape[0] / 10) * 9),
             int(slice.shape[1] / 10):int((slice.shape[1] / 10) * 9)]
    centerMean = [center[:, :, 0].mean(), center[:, :, 1].mean(), center[:, :, 2].mean()]

    topBorder = slice[0:int(slice.shape[0] / 10), :]
    botBorder = slice[int(slice.shape[0] - slice.shape[0] / 10):slice.shape[0], :]
    leftBorder = slice[int(slice.shape[0] / 10):int(slice.shape[0] - slice.shape[0] / 10), :int(slice.shape[1] / 10)]
    rightBorder = slice[int(slice.shape[0] / 10):int(slice.shape[0] - slice.shape[0] / 10),
                   int(slice.shape[1] - slice.shape[1] / 10):slice.shape[1]]

    border_mean = [(topBorder[:, :, 0].mean() + botBorder[:, :, 0].mean()
                        + leftBorder[:, :,0].mean() + rightBorder[:, :,0].mean() / 4),
                   ((topBorder[:, :, 1]).mean() + botBorder[:, :, 1].mean()
                        + leftBorder[:, :, 1].mean() + rightBorder[:, :,1].mean() / 4),
                   ((topBorder[:, :, 2].mean() + botBorder[:, :, 2]).mean()
                        + leftBorder[:, :,2].mean() + rightBorder[:, :,2].mean() / 4)]

    cutSlice = [centerMean, border_mean]

    return cutSlice


def getType(slice, name):
    data = loadTileVectorDictionary()
    type, distance = singleNearestNeighbor(slice, data)
    return [type, distance]


def evenLightingSQRT(image):
    output = image.copy().astype("float64")
    # while output.mean() < 125:
    for channel in range(image.shape[2]):
        # output[:,:,channel] /= 255
        # output[:, :, channel] = np.sqrt(output[:,:,channel])
        # output[:, :, channel] *= 255
        output[:, :, channel] = np.sqrt(output[:, :, channel])

    return output.astype("uint8")


def evenLightingCubed(image):
    output = image.astype("float64")
    for channel in range(image.shape[2]):
        output[:, :, channel] = output[:, :, channel] ** 2
    return output.astype("uint8")


def calculateEuclidianDistance(feature_vector1, feature_vector2):
    dist = np.sqrt(np.sum((feature_vector1 - feature_vector2) ** 2))
    return dist


def calculateImageHistogramBinVector(image, bins: int, name):
    def showHist(figName):
        B_h, B_bins = np.histogram(image[:, :, 0], bins, [0, 256])
        G_h, G_bins = np.histogram(image[:, :, 1], bins, [0, 256])
        R_h, R_bins = np.histogram(image[:, :, 2], bins, [0, 256])
        fig = plt.figure()
        fig.suptitle(figName, fontsize=15)
        width = 0.7 * (B_bins[1] - B_bins[0])
        plt.subplot(1, 3, 1)
        plt.ylim([0, 10000])
        plt.title('Blue')
        plt.bar(B_bins[:-1], B_h, width=width, color='blue', label='Blue')
        plt.subplot(1, 3, 2)
        plt.ylim([0, 10000])
        plt.title('Green')
        plt.bar(G_bins[:-1], G_h, width=width, color='green', label='Green')
        plt.subplot(1, 3, 3)
        plt.ylim([0, 10000])
        plt.title('Red')
        plt.bar(R_bins[:-1], R_h, width=width, color='red', label='Red')
        fig.show()

    # Vi laver et histrogram til hver farvekanal
    B_hist = np.histogram(image[:, :, 0], bins, [0, 256])
    G_hist = np.histogram(image[:, :, 1], bins, [0, 256])
    R_hist = np.histogram(image[:, :, 2], bins, [0, 256])

    # Vi fyrer alle histogrammer ind i røven ad hinanden i et ny array 'hist' for at få det som en feature vektor
    hist = np.concatenate((B_hist[0], G_hist[0], R_hist[0]))

    # Hvis vi vil vise figuren for histogrmammet
    # showHist(name)

    def createHistVector(hist):
        histMax = max(hist)

        if list(hist).index(histMax) == 0:
            histIndex = 1
        elif list(hist).index(histMax) == binsize - 1:
            histIndex = binsize - 2
        else:
            histIndex = list(hist).index(histMax)

        histLower = histIndex - 1
        histUpper = histIndex + 1

        indexDescriptor = (hist[histIndex] + hist[histLower] + hist[histUpper]) / np.sum(hist)

        histVector = [indexDescriptor, histLower, histIndex, histUpper]

        return histVector

    return [createHistVector(B_hist[0]), createHistVector(G_hist[0]), createHistVector(R_hist[0])]


def saveTileVectorDictionary():
    featureVectors = {}

    for tileType in tileTypes:
        tileArray = []
        for rotation in range(4):
            for tile in range(10):
                img = cv.imread(f'King Domino dataset/Cropped_Tiles/{tileType}/{rotation}/{tile}.jpg')
                lightCorrectedImg = evenLightingSQRT(img)
                diff = calculateDiffereneOfGaussian(img)
                gaussianCenterMean, gaussianBorderMean = defineCenterAndBorder(diff)
                centerMean, borderMean = defineCenterAndBorder(lightCorrectedImg)
                tileArray.append(
                    [np.array(calculateSliceColor_Mean(lightCorrectedImg)), np.array(centerMean), np.array(borderMean), np.array(gaussianCenterMean)])
        featureVectors[tileType] = tileArray

    np.save(f'King Domino dataset/Cropped_Tiles/featureVectors', featureVectors)


def loadTileVectorDictionary():
    featureVectors = np.load(f'King Domino dataset/Cropped_Tiles/featureVectors.npy', allow_pickle=True).tolist()
    return featureVectors



def calculateBinIndexDistance(sliceBINdexVector, data):
    # Find distance mellem descriptors

    # Find vægt mellem indicer
    weightArraySlice = []
    weightArrayData = []

    for i, vector in enumerate(sliceBINdexVector):
        numbersInCommon = 1
        for index in range(1, len(vector)):
            if index in data[i]:
                numbersInCommon -= 0.3
        weightArraySlice.append(numbersInCommon * vector[0])
        weightArrayData.append(numbersInCommon * data[i][0])

    return math.sqrt(
        (weightArraySlice[0] - weightArrayData[0]) ** 2 + (weightArraySlice[1] - weightArrayData[1]) ** 2 + (
                weightArraySlice[2] - weightArrayData[2]) ** 2)


def calculateMedianDistance(slice, data):
    return math.sqrt((slice[0] - data[0]) ** 2 + (slice[1] - data[1]) ** 2 + (slice[2] - data[2]) ** 2)


def calculateModeDistance(slice, data):
    return math.sqrt((slice[0] - data[0]) ** 2 + (slice[1] - data[1]) ** 2 + (slice[2] - data[2]) ** 2)


def calculateEuclidianDist(slice, data):
    meanDistance = ((slice[0][0] - data[0][0]) ** 2 + (slice[0][1] - data[0][1]) ** 2 + (slice[0][2] - data[0][2]) ** 2)
    centerDistance = ((slice[1][0] - data[1][0]) ** 2 + (slice[1][1] - data[1][1]) ** 2 + (slice[1][2] - data[1][2]) ** 2)
    borderDistance = ((slice[2][0] - data[2][0]) ** 2 + (slice[2][1] - data[2][1]) ** 2 + (slice[2][2] - data[2][2]) ** 2)
    gausDistance = ((slice[3][0] - data[3][0]) ** 2 + (slice[3][1] - data[3][1]) ** 2 + (slice[3][2] - data[3][2]) ** 2)
    return math.sqrt(meanDistance+centerDistance+borderDistance)

def singleNearestNeighbor(slice, data):
    diff = calculateDiffereneOfGaussian(evenLightingCubed(slice))
    gaussianCenterMean, gaussianBorderMean = defineCenterAndBorder(diff)
    centerMean, borderMean = defineCenterAndBorder(slice)
    sliceFeatureVector = [np.array(calculateSliceColor_Mean(slice)), np.array(centerMean), np.array(borderMean),np.array(gaussianCenterMean)]

    distance, currentType = 6, 'None_0'

    for (tileType, tiles) in data.items():
        for tile in tiles:
            newDistance = calculateEuclidianDist(sliceFeatureVector, tile)
            if (newDistance < distance):
                distance = newDistance
                currentType = tileType
    #currentType, distance = calcCrowns(currentType, sliceFeatureVector, data)
    return [currentType, distance]


def kNearestNeighbor(slice, data):
    diff = calculateDiffereneOfGaussian(slice)
    gaussianMean = calculateGaussianMean(diff)
    centerMean, borderMean = defineCenterAndBorder(slice)
    sliceFeatureVector = [np.array(calculateSliceColor_Mean(slice)), np.array(centerMean), np.array(borderMean),
                          np.array(gaussianMean)]

    th = 20  # threshold
    distanceArray = [th, th, th, th, th]
    tileTypeArray = ['None_0', 'None_0', 'None_0', 'None_0', 'None_0']

    for (tileType, tiles) in data.items():
        for tile in tiles:
            newDistance = calculateEuclidianDist(sliceFeatureVector, tile)
            for i, score in enumerate(distanceArray):
                if (newDistance < score):
                    distanceArray[i] = newDistance
                    tileTypeArray[i] = tileType
                    break
    print(tileTypeArray)
    tileType = mode(tileTypeArray)
    distance = distanceArray[tileTypeArray.index(tileType)]

    return [tileType, distance]


def calculateSliceColor_Mode(slice):
    b = stats.mode(slice[:, :, 0], axis=None)[0]
    g = stats.mode(slice[:, :, 1], axis=None)[0]
    r = stats.mode(slice[:, :, 2], axis=None)[0]
    return [b, g, r]


def calculateSliceColor_Median(slice):
    b = np.median(slice[:, :, 0])
    g = np.median(slice[:, :, 1])
    r = np.median(slice[:, :, 2])
    return [b, g, r]

def calculateGaussianMean(slice):
    b = []
    g =[]
    r = []
    for y, row in enumerate(slice):
        for x, pixel in enumerate(row):
            if t.calculateIntensity(pixel) > 0.15:
                b.append(pixel[0])
                g.append(pixel[1])
                r.append(pixel[2])
    if len(b) > 0:
        b = (normalize(np.array(b))*255).mean()
    else:
        b = 0
    if len(g) > 0:
        g = (normalize(np.array(g))*255).mean()
    else:
        g = 0
    if len(r) > 0:
        r = (normalize(np.array(r))*255).mean()
    else:
        r = 0
    return [b,g,r]
def normalize(array):
    return (array - np.min(array))/np.ptp(array)
def calculateSliceColor_Mean(slice):

    bMean = slice[:, :, 0].mean()
    gMean = slice[:, :, 1].mean()
    rMean = slice[:, :, 2].mean()


    return [bMean, gMean, rMean]


def getScore(sliceTypes):
    def startBurning(startPos, burningImage):
        burnQueue = deque()
        currentIsland = []
        burnQueue.append(startPos)
        while (len(burnQueue) > 0):
            nextpos = burnQueue.pop()
            currentIsland.append([nextpos[0], nextpos[1]])
            if (nextpos[0] - 1 >= 0 and [nextpos[0] - 1, nextpos[1]] not in currentIsland and [nextpos[0] - 1, nextpos[
                1]] not in burnQueue and burningImage[nextpos[0]][nextpos[1]].split("_")[0] ==
                    burningImage[nextpos[0] - 1][nextpos[1]].split("_")[0]):
                burnQueue.append([nextpos[0] - 1, nextpos[1]])
            if (nextpos[0] + 1 <= 4 and [nextpos[0] + 1, nextpos[1]] not in currentIsland and [nextpos[0] + 1, nextpos[
                1]] not in burnQueue and burningImage[nextpos[0]][nextpos[1]].split("_")[0] ==
                    burningImage[nextpos[0] + 1][nextpos[1]].split("_")[0]):
                burnQueue.append([nextpos[0] + 1, nextpos[1]])
            if (nextpos[1] - 1 >= 0 and [nextpos[0], nextpos[1] - 1] not in currentIsland and [nextpos[0], nextpos[
                                                                                                               1] - 1] not in burnQueue and
                    burningImage[nextpos[0]][nextpos[1]].split("_")[0] ==
                    burningImage[nextpos[0]][nextpos[1] - 1].split("_")[0]):
                burnQueue.append([nextpos[0], nextpos[1] - 1])
            if (nextpos[1] + 1 <= 4 and [nextpos[0], nextpos[1] + 1] not in currentIsland and [nextpos[0], nextpos[
                                                                                                               1] + 1] not in burnQueue and
                    burningImage[nextpos[0]][nextpos[1]].split("_")[0] ==
                    burningImage[nextpos[0]][nextpos[1] + 1].split("_")[0]):
                burnQueue.append([nextpos[0], nextpos[1] + 1])
        return currentIsland

    ArrayOfIslands = []
    for y, sliceTypeRow in enumerate(sliceTypes):
        for x, sliceType in enumerate(sliceTypes):
            found = False
            for island in ArrayOfIslands:
                if [y, x] in island:
                    found = True
                    break
            if not found:
                ArrayOfIslands.append(startBurning([y, x], sliceTypes))
    ArrayOfIslandsAsCrowns = []
    for currentIsland in ArrayOfIslands:
        currentIslandAsCrowns = []
        for pos in currentIsland:
            currentIslandAsCrowns.append(int(sliceTypes[pos[0]][pos[1]].split("_")[1]))
        ArrayOfIslandsAsCrowns.append(currentIslandAsCrowns)

    score = 0
    print(f'crownArray:\n{ArrayOfIslandsAsCrowns}')
    for island in ArrayOfIslandsAsCrowns:
        score += len(island) * np.sum(island)

    return score
def calcCrowns(sliceType, slice, data):
    distance = 100
    currentSliceType = sliceType
    for (tileType, tiles) in data.items():
        if tileType.split("_")[0] == sliceType.split("_")[0]:
            for tile in tiles:
                newDistance = math.sqrt((slice[3][0] - tile[3][0]) ** 2 + (slice[3][1] - tile[3][1]) ** 2 + (
                            slice[3][2] - tile[3][2]) ** 2)
                if newDistance < distance:
                    currentSliceType = tileType
                    distance = newDistance
    return [currentSliceType,distance]
def getAllSliceTypes(slices, data):
    sliceTypes = []
    distances = []

    for y, row in enumerate(slices):
        sliceTypeRow = []
        distanceRow = []

        for x, slice in enumerate(row):
            sliceType, distance = singleNearestNeighbor(slice, data)
            sliceTypeRow.append(sliceType)
            distanceRow.append(distance)
        sliceTypes.append(sliceTypeRow)
        distances.append(distanceRow)

    return [sliceTypes, distances]


def addTileText(image, slices, sliceTypes, distances):
    outputImage = image.copy()

    for y, row in enumerate(slices):
        for x, slice in enumerate(row):
            xCoord = int((x * outputImage.shape[1] / 5) + 5)
            yCoord = int((y * outputImage.shape[0] / 5) + outputImage.shape[0] / 20)

            cv.putText(outputImage, f'{sliceTypes[y][x]}:', (xCoord, yCoord + 00), cv.FONT_HERSHEY_PLAIN, 1,
                       (255, 255, 255))
            cv.putText(outputImage, f'{distances[y][x]:.2f}', (xCoord, yCoord + 15), cv.FONT_HERSHEY_PLAIN, 1,
                       (255, 255, 255))

            # cv.imshow(f'{y, x}', slice)

    return outputImage


def calculateDiffereneOfGaussian(image):
    borderRoi = b.addborder_reflect(image, 7)
    kernel = g.makeGuassianKernel(7, 1.6)
    blurredRoi = g.convolve(borderRoi, kernel)
    differenceImg = cv.subtract(image, blurredRoi)

    # for channel in range(differenceImg.shape[2]):
    #     arrayToNormalize = differenceImg[:,:,channel].astype("float64")
    #     differenceImg[:,:,channel] = (normalize(arrayToNormalize)*255).astype("uint8")
    return differenceImg



### Main method call ###

if __name__ == "__main__":
    print("# LOADING IMAGE : ", end = '')
    st = time.time()
    # saveTileVectorDictionary()
    for i in range (59,74):
        ROI = returnSingleImage(gameboardList, i)
        print(f'{time.time()-st:.4f} s')

        print("# CORRECTING IMAGE DATA : ", end = '')
        st = time.time()
        evenROI = evenLightingSQRT(ROI)
        print(f'{time.time()-st:.4f} s')

        # print("# CALCLUTATING DIFFERENCE OF GAUSSIAN : ", end = '')
        # st = time.time()
        # diff = calculateDiffereneOfGaussian(ROI)
        # cv.imshow(f'diff {i}', diff)
        # print(f'{time.time()-st:.4f} s')
        #
        # print("# THRESHOLDING GAUSSIAN DIFFERENCE : ", end = '')
        # st = time.time()
        # tresholdedROI = t.makeImageBinaryIntensityThreshold(diff, 0.15)
        # print(f'{time.time()-st:.4f} s')
        #
        # print("# PERFORMING MORPHOLOGY ON BINARY IMAGE : ", end = '')
        # st = time.time()
        # closed = m.close(tresholdedROI,3)
        # eroded = m.erode(tresholdedROI,3)
        #
        # cv.imshow(f'threshold{i}', tresholdedROI)
        # cv.imshow(f'closed{i}', closed)
        # cv.imshow(f'eroded {i}', eroded)
        # print(f'{time.time()-st:.4f} s')

        print("# LOADING DATA : ", end = '')
        st = time.time()

        data = loadTileVectorDictionary()
        print(f'{time.time()-st:.4f} s')

        print("# SLICING IMAGE : ", end = '')
        st = time.time()
        sliceArray = sliceROI(evenROI)
        print(f'{time.time()-st:.4f} s')

        print("# DECLARING SLICE TYPES : ", end = '')
        st = time.time()
        sliceTypeArray, distanceArray = getAllSliceTypes(sliceArray, data)
        print(f'{time.time()-st:.4f} s')

        print("# COMPUTING SCORE")
        score = getScore(sliceTypeArray)
        print(f'totalScore image {i}:\n{score}')
        print("# ADDING TEXT TO IMAGE : ", end = '')
        st = time.time()
        ROI_text = addTileText(ROI, sliceArray, sliceTypeArray, distanceArray)
        print(f'{time.time()-st:.4f} s')

        print("# SHOWING IMAGE : ", end = '')
        st = time.time()
        cv.imshow(f'Gameboard NR: {i}', ROI_text)
        print(f'{time.time()-st:.4f} s')

    cv.waitKey(0)
