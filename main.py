from collections import deque
import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy import stats
from statistics import mode

gameboardList = []
gameareasList = []

tileTypes = ['water_0', 'water_1', 'desert_0', 'desert_1', 'forest_0', 'forest_1', 'grass_0', 'grass_1',
             'grass_2', 'mine_0', 'mine_1', 'mine_2', 'mine_3', 'waste_0', 'waste_1', 'waste_2']

binsize = 8


def add_boards():
    temp_list = []
    for i in range(1, 75):
        pic_path = 'King Domino dataset/Cropped and perspective corrected boards/' + str(i) + '.jpg'
        pic_temp = cv.imread(pic_path)
        temp_list.append(pic_temp)
    return temp_list


def add_areas():
    temp_list = []
    for i in range(1, 19):
        pic_path = 'King Domino dataset/Full game areas/DSC_' + str(1262 + i) + '.JPG'
        pic_temp = cv.imread(pic_path)
        temp_list.append(pic_temp)
    return temp_list


gameboardList.extend(add_boards())
gameareasList.extend(add_areas())


def show_list_of_images(list):
    for i in range(len(list)):
        cv.imshow(str(i + 1), list[i])
    cv.waitKey(0)


def show_single_image(list, index):
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


def return_single_image(list, index):
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
    center = slice[int(slice.shape[0] / 8):int((slice.shape[0] / 8) * 7),
             int(slice.shape[1] / 8):int((slice.shape[1] / 8) * 7)]
    centerMean = [int(center[:, :, 0].mean()), int(center[:, :, 1].mean()), int(center[:, :, 2].mean())]

    top_border = slice[0:int(slice.shape[0] / 8), :]
    bot_border = slice[int(slice.shape[0] - slice.shape[0] / 8):slice.shape[0], :]
    left_border = slice[int(slice.shape[0] / 8):int(slice.shape[0] - slice.shape[0] / 8), 0:int(slice.shape[1] / 8)]
    right_border = slice[int(slice.shape[0] / 8):int(slice.shape[0] - slice.shape[0] / 8),
                   int(slice.shape[1] - slice.shape[1] / 8):slice.shape[1]]

    border_mean = [int((top_border[:, :, 0].mean() + bot_border[:, :, 0].mean() + left_border[:, :,
                                                                                  0].mean() + right_border[:, :,
                                                                                              0].mean()) / 4),
                   int((top_border[:, :, 1].mean() + bot_border[:, :, 1].mean() + left_border[:, :,
                                                                                  1].mean() + right_border[:, :,
                                                                                              1].mean()) / 4),
                   int((top_border[:, :, 2].mean() + bot_border[:, :, 2].mean() + left_border[:, :,
                                                                                  2].mean() + right_border[:, :,
                                                                                              2].mean()) / 4)]

    cut_slice = [centerMean, border_mean]

    return cut_slice


def getType(slice, name):
    data = loadTileVectorDictionary()
    type, distance = kNearestNeighbor(slice, data)
    return [type, distance]


def calculateEuclidianDistance(feature_vector1, feature_vector2):
    dist = np.sqrt(np.sum((feature_vector1 - feature_vector2) ** 2))
    return dist


def calculateImageHistogramBinVector(image, bins: int, name):
    def showHist(fig_name):
        B_h, B_bins = np.histogram(image[:, :, 0], bins, [0, 256])
        G_h, G_bins = np.histogram(image[:, :, 1], bins, [0, 256])
        R_h, R_bins = np.histogram(image[:, :, 2], bins, [0, 256])
        fig = plt.figure()
        fig.suptitle(fig_name, fontsize=15)
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
        hist_max = max(hist)

        if list(hist).index(hist_max) == 0:
            hist_index = 1
        elif list(hist).index(hist_max) == binsize - 1:
            hist_index = binsize - 2
        else:
            hist_index = list(hist).index(hist_max)

        hist_lower = hist_index - 1
        hist_upper = hist_index + 1

        indexDescriptor = (hist[hist_index] + hist[hist_lower] + hist[hist_upper]) / np.sum(hist)

        hist_vector = [indexDescriptor, hist_lower, hist_index, hist_upper]

        return hist_vector


    return [createHistVector(B_hist[0]), createHistVector(G_hist[0]), createHistVector(R_hist[0])]


def saveTileVectorDictionary():
    featureVectors = {}

    for tileType in tileTypes:
        tileArray = []
        for tile in range(10):
            img = cv.imread(f'King Domino dataset/Cropped_Tiles/{tileType}/{tile}.jpg')
            tileArray.append(np.array(calculateSliceColor_Mean(img)))
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
    return math.sqrt((slice[0] - data[0]) ** 2 + (slice[1] - data[1]) ** 2 + (slice[2] - data[2]) ** 2)


def kNearestNeighbor(slice, data):
    sliceFeatureVector = calculateSliceColor_Mean(slice)

    distance, currentType = 20.0, 'None_0'

    for (tileType, tiles) in data.items():
        for tile in tiles:
            newDistance = calculateEuclidianDist(sliceFeatureVector, tile)
            if (newDistance < distance):
                distance = newDistance
                currentType = tileType

    return [currentType, distance]


def kNearestNeighbor_old(slice, data):
    meanVector = calculateSliceColor_Mean(slice)

    th = 20  # threshold
    distanceArray = [th, th, th, th, th]
    tileTypeArray = ['None_0', 'None_0', 'None_0', 'None_0', 'None_0']

    for (tileType, tiles) in data.items():
        for tile in tiles:
            new_distance = calculateEuclidianDist(meanVector, tile)
            for i, score in enumerate(distanceArray):
                if (new_distance < score):
                    distanceArray[i] = new_distance
                    tileTypeArray[i] = tileType
                    break

    tileType = tileTypeArray[0]
    distance = distanceArray[0]

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


def calculateSliceColor_Mean(slice):
    b_mean = slice[:, :, 0].mean()
    g_mean = slice[:, :, 1].mean()
    r_mean = slice[:, :, 2].mean()

    return [b_mean, g_mean, r_mean]


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
    print(ArrayOfIslandsAsCrowns)
    for island in ArrayOfIslandsAsCrowns:
        score += len(island) * np.sum(island)
    print(score)
    return score


def getAllSliceTypes(slices, data):
    sliceTypes = []
    distances = []

    for y, row in enumerate(slices):
        sliceTypeRow = []
        distanceRow = []

        for x, slice in enumerate(row):
            sliceType, distance = kNearestNeighbor(slice, data)
            sliceTypeRow.append(sliceType)
            distanceRow.append(distance)
        sliceTypes.append(sliceTypeRow)
        distances.append(distanceRow)

    return [sliceTypes, distances]


def addTileText(image, slices, sliceTypes, distances):
    output_image = image.copy()

    for y, row in enumerate(slices):
        for x, slice in enumerate(row):
            x_coord = int((x * output_image.shape[1] / 5) + 5)
            y_coord = int((y * output_image.shape[0] / 5) + output_image.shape[0] / 20)

            cv.putText(output_image, f'{sliceTypes[y][x]}:', (x_coord, y_coord + 00), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
            cv.putText(output_image, f'{int(distances[y][x])}', (x_coord, y_coord + 15), cv.FONT_HERSHEY_PLAIN, 1,
                       (255, 255, 255))

            # cv.imshow(f'{y, x}', slice)

    return output_image


### Main method call ###

if __name__ == "__main__":
    # INDLÆS DET BILLEDE SOM VI VIL ARBEJDE MED
    print("Loading Picture")
    ROI = return_single_image(gameboardList, 59)

    # INDLÆS DATA
    print("Loading Data")
    #saveTileVectorDictionary()
    data = loadTileVectorDictionary()

    # OPRET ARRAY AF SLICES FRA GIVNE SPILLEPLADEBILLEDE
    print("Slicing Roi")
    sliceArray = sliceROI(ROI)

    print("Distinguishing Types")
    # BESTEM ET ARRAY MED ALLE SLICES TILE-TYPER I MED TILSVARENDE KOORDINATER
    sliceTypeArray, distanceArray = getAllSliceTypes(sliceArray, data)
    print("Computing score")
    score = getScore(sliceTypeArray)
    print(score)
    print("Adding text")
    # TILFØJ TEKST TIL ET OUTPUT BILLEDE
    ROI_text = addTileText(ROI, sliceArray, sliceTypeArray, distanceArray)
    print("Showing image")
    # VIS BILLEDE
    cv.imshow('ROI_with_text', ROI_text)
    #cv.imshow('ROI',ROI)
    cv.waitKey(0)
