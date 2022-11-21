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


def slice_roi(roi):
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
    data = createTileFeaturevectorArray()
    type, distance = kNearestNeighbor(slice, data)

    """


    distance = [48, 48, 48, 10000]
    type = ['None', 'None', 'None', 'Nej']

    meanArray = [[150, 80, 14, 'wp'], [113, 80, 43, 'w1k'],
                 [22, 60, 46, 'fp'], [27, 62, 57, 'f1k'],
                 [7, 155, 179, 'dp'], [19, 142, 169, 'd1k'],
                 [26, 51, 59, 'mp'], [27, 68, 77, 'm1k'], [26, 51, 61, 'm2k'], [33, 62, 73, 'm3k'],
                 [60, 100, 111, 'wlp'], [54, 93, 104, 'wl1k'], [48, 89, 103, 'wl2k'],
                 [28, 146, 100, 'gp'], [25, 116, 103, 'g1k'], [34, 126, 114, 'g2k']]

    borderMeanArray = [[136, 79, 23, 'wp'], [126, 73, 24, 'w1k'],
                       [23, 72, 61, 'fp'], [22, 67, 59, 'f1k'],
                       [15, 160, 180, 'dp'], [16, 151, 174, 'd1k'],
                       [19, 37, 43, 'mp'], [26, 38, 42, 'm1k'], [22, 40, 47, 'm2k'], [25, 39, 46, 'm3k'],
                       [56, 102, 114, 'wlp'], [58, 102, 112, 'wl1k'], [50, 98, 112, 'wl2k'],
                       [31, 140, 100, 'gp'], [25, 127, 91, 'g1k'], [32, 137, 105, 'g2k']]

    centerMeanArray = [[162, 79, 5, 'wp'], [97, 81, 54, 'w1k'],
                       [18, 43, 28, 'fp'], [29, 57, 55, 'f1k'],
                       [9, 172, 194, 'dp'], [21, 135, 161, 'd1k'],
                       [34, 54, 61, 'mp'], [34, 58, 67, 'm1k'], [31, 67, 83, 'm2k'], [38, 73, 91, 'm3k'],
                       [58, 93, 103, 'wlp'], [50, 86, 98, 'wl1k'], [45, 83, 96, 'wl2k'],
                       [27, 151, 98, 'gp'], [24, 103, 100, 'g1k'], [37, 111, 122, 'g2k']]


    for i, tile_type in enumerate(meanArray):
        new_distance = math.sqrt((B - tile_type[0]) ** 2 + (G - tile_type[1]) ** 2 + (R - tile_type[2]) ** 2)
        if new_distance < distance[0]:
            distance[0] = new_distance
            type[0] = tile_type[3]

    for i, tile_type in enumerate(borderMeanArray):
        new_distance = math.sqrt((B - tile_type[0]) ** 2 + (G - tile_type[1]) ** 2 + (R - tile_type[2]) ** 2)
        if new_distance < distance[1]:
            distance[1] = new_distance
            type[1] = tile_type[3]

    for i, tile_type in enumerate(centerMeanArray):
        new_distance = math.sqrt((B - tile_type[0]) ** 2 + (G - tile_type[1]) ** 2 + (R - tile_type[2]) ** 2)
        if new_distance < distance[2]:
            distance[2] = new_distance
            type[2] = tile_type[3]

    sliceVector = calculateImageHistogramBinVector(slice, binsize, name)
    # distance = 300
    # type[3] = 'Nej'

    for tileType, featureVector in feature_vectors.items():
        new_distance = calculateEuclidianDistance(sliceVector, featureVector)
        # print(new_distance)
        if new_distance < distance[3]:
            distance[3] = new_distance
            type[3] = tileType
    """

    return [type, distance]


def calculateEuclidianDistance(feature_vector1, feature_vector2):
    dist = np.sqrt(np.sum((feature_vector1 - feature_vector2) ** 2))
    return dist


def calculateImageHistogramBinVector(image, bins: int, name, factor: int = 255):
    # Vi laver et histrogram til hver farvekanal
    B_hist = np.histogram(image[:, :, 0], bins, [0, 256])
    G_hist = np.histogram(image[:, :, 1], bins, [0, 256])
    R_hist = np.histogram(image[:, :, 2], bins, [0, 256])

    # Vi fyrer alle histogrammer ind i røven ad hinanden i et ny array 'hist' for at få det som en feature vektor
    hist = np.concatenate((B_hist[0], G_hist[0], R_hist[0]))

    B_h, B_bins = np.histogram(image[:, :, 0], bins, [0, 256])
    G_h, G_bins = np.histogram(image[:, :, 1], bins, [0, 256])
    R_h, R_bins = np.histogram(image[:, :, 2], bins, [0, 256])

    # fig = plt.figure()
    # fig.suptitle(name, fontsize=15)
    # width = 0.7 * (B_bins[1] - B_bins[0])
    # plt.subplot(1, 3, 1)
    # plt.ylim([0, 10000])
    # plt.title('Blue')
    # plt.bar(B_bins[:-1], B_h, width=width, color='blue', label='Blue')
    # plt.subplot(1, 3, 2)
    # plt.ylim([0, 10000])
    # plt.title('Green')
    # plt.bar(G_bins[:-1], G_h, width=width, color='green', label='Green')
    # plt.subplot(1, 3, 3)
    # plt.ylim([0, 10000])
    # plt.title('Red')
    # plt.bar(R_bins[:-1], R_h, width=width, color='red', label='Red')
    # fig.show()

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

    # normaliserer histogrammet således at værdierne ligger mellem 0 og 1
    """
    hist = hist.astype(np.float64)
    if (hist.max() != 0 or None):
        hist /= int(hist.max())
        hist *= factor
    """

    return [createHistVector(B_hist[0]), createHistVector(G_hist[0]), createHistVector(R_hist[0])]


def saveTileVectors():
    def saveVector(fileName, tileType):
        img = cv.imread(f'King Domino dataset/Cropped_Tiles/{tileType}/{fileName}.jpg')
        np.save(f'King Domino dataset/Cropped_Tiles/{tileType}/{fileName}',
                calculateSliceColor_Mean(img))

    for tileType in tileTypes:
        for tile in range(10):
            saveVector(tile, tileType)


def calculateBinIndexDistance(sliceBINdexVector, data):
    # Find distance mellem descriptors

    # Find vægt mellem indicer
    weightArraySlice = []
    weightArrayData = []

    for i, vector in enumerate(sliceBINdexVector):
        numbersInCommon = 1
        for index in range(1,len(vector)):
            if index in data[i]:
                numbersInCommon -= 0.3
        weightArraySlice.append(numbersInCommon*vector[0])
        weightArrayData.append(numbersInCommon*data[i][0])

    return math.sqrt((weightArraySlice[0]-weightArrayData[0])**2+(weightArraySlice[1]-weightArrayData[1])**2+(weightArraySlice[2]-weightArrayData[2])**2)


def calculateMedianDistance(slice, data):
    return math.sqrt((slice[0] - data[0]) ** 2 + (slice[1] - data[1]) ** 2 + (slice[2] - data[2]) ** 2)
def calculateModeDistance(slice, data):
    return math.sqrt((slice[0] - data[0]) ** 2 + (slice[1] - data[1]) ** 2 + (slice[2] - data[2]) ** 2)
def calculateMeanDistance(slice, data):
    return math.sqrt((slice[0] - data[0]) ** 2 + (slice[1] - data[1]) ** 2 + (slice[2] - data[2]) ** 2)
def kNearestNeighbor(slice, data):
    sliceHist = calculateSliceColor_Mean(slice)

    tileTypeArray = ['None_0', 'None_0', 'None_0', 'None_0', 'None_0']
    distanceArray = [20, 20, 20, 20, 20]

    for (tileType, tiles) in data.items():
        for tile in tiles:
            new_distance = calculateMeanDistance(sliceHist, tile)
            for i, score in enumerate(distanceArray):
                if (new_distance < score):
                    distanceArray[i] = new_distance
                    tileTypeArray[i] = tileType
                    break

    tileType = tileTypeArray[0]
    distance = distanceArray[0]

    return [tileType, distance]


def createTileFeaturevectorArray():
    featureVectors = {}

    for tileType in tileTypes:
        tileArray = []
        for tile in range(10):
            tileArray.append(np.load(f'King Domino dataset/Cropped_Tiles/{tileType}/{tile}.npy', allow_pickle=True))
        featureVectors[tileType] = tileArray

    return featureVectors


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
    b = slice[:, :, 0].mean()
    g = slice[:, :, 1].mean()
    r = slice[:, :, 2].mean()
    return [b, g, r]

def getScore(sliceTypes):
    def startBurning(startPos, burningImage):
        burnQueue = deque()
        currentIsland = []
        burnQueue.append(startPos)
        while(len(burnQueue) > 0):
            nextpos = burnQueue.pop()
            currentIsland.append([nextpos[0], nextpos[1]])
            if(nextpos[0]- 1 >= 0 and [nextpos[0]- 1,nextpos[1]] not in currentIsland and [nextpos[0]- 1,nextpos[1]] not in burnQueue and burningImage[nextpos[0]][nextpos[1]].split("_")[0] ==
                    burningImage[nextpos[0]-1][nextpos[1]].split("_")[0]):
                burnQueue.append([nextpos[0] - 1, nextpos[1]])
            if (nextpos[0] + 1 <= 4 and [nextpos[0]+ 1,nextpos[1]] not in currentIsland and [nextpos[0]+ 1,nextpos[1]] not in burnQueue and burningImage[nextpos[0]][nextpos[1]].split("_")[0] ==
                    burningImage[nextpos[0] + 1][nextpos[1]].split("_")[0]):
                burnQueue.append([nextpos[0] + 1, nextpos[1]])
            if (nextpos[1] - 1 >= 0 and [nextpos[0],nextpos[1]-1] not in currentIsland and [nextpos[0],nextpos[1]-1] not in burnQueue and burningImage[nextpos[0]][nextpos[1]].split("_")[0] ==
                    burningImage[nextpos[0]][nextpos[1]-1].split("_")[0]):
                burnQueue.append([nextpos[0], nextpos[1] - 1])
            if (nextpos[1] + 1 <= 4 and [nextpos[0],nextpos[1] + 1] not in currentIsland and [nextpos[0],nextpos[1]+1] not in burnQueue and burningImage[nextpos[0]][nextpos[1]].split("_")[0] ==
                    burningImage[nextpos[0]][nextpos[1] + 1].split("_")[0]):
                burnQueue.append([nextpos[0], nextpos[1]+1])
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
                ArrayOfIslands.append(startBurning([y,x], sliceTypes))
    ArrayOfIslandsAsCrowns = []
    for currentIsland in ArrayOfIslands:
        currentIslandAsCrowns = []
        for pos in currentIsland:
            currentIslandAsCrowns.append(int(sliceTypes[pos[0]][pos[1]].split("_")[1]))
        ArrayOfIslandsAsCrowns.append(currentIslandAsCrowns)

    score = 0
    print(ArrayOfIslandsAsCrowns)
    for island in ArrayOfIslandsAsCrowns:
        score += len(island)*np.sum(island)
    print(score)
    return score
############################################ Method calls


saveTileVectors()
# print(createTileFeaturevectorArray())

ROI = return_single_image(gameboardList, 54)
# mROI = cv.medianBlur(ROI, 5)
slices = slice_roi(ROI)
sliceTypes = []
for y, row in enumerate(slices):
    sliceTypeRow = []
    for x, slice in enumerate(row):
        sliceType, distance = getType(slice, f'{y, x}')
        sliceTypeRow.append(sliceType)
        # print(calculateSliceColor_Median(slice))
        # print(calculateSliceColor_Mode(slice))

        x_coord = int((x * ROI.shape[1] / 5) + 5)
        y_coord = int((y * ROI.shape[0] / 5) + ROI.shape[0] / 20)

        cv.putText(ROI, f'{sliceType}:', (x_coord, y_coord + 00), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        cv.putText(ROI, f'{int(distance)}', (x_coord, y_coord + 15), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        cv.imshow(f'{y, x}', slice)
        # cv.putText(ROI, f'{sliceType[1]}: {int(distance)}', (x_coord, y_coord + 15), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        # cv.putText(ROI, f'{sliceType[2]}: {int(distance)}', (x_coord, y_coord + 30), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        # cv.putText(ROI, f'{sliceType[3]}: {int(distance)}', (x_coord, y_coord + 45), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        # print(f'({y, x}). (BGR):{int(slices[y][x][:, :, 0].mean()), int(slices[y][x][:, :, 1].mean()), int(slices[y][x][:, :, 2].mean())}. (Center,Border): {defineCenterAndBorder(slice)}')
    sliceTypes.append(sliceTypeRow)
print(sliceTypes)
score = getScore(sliceTypes)
cv.imshow('Roi_with_contours', ROI)

# cv.imshow('Slice', slices[4][2])
cv.waitKey(0)
