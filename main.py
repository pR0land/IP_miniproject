import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

gameboard_list = []
gameareas_list = []

tile_types = ['wp', 'w1k', 'dp', 'd1k', 'fp', 'f1k', 'gp', 'g1k', 'g2k', 'mp', 'm1k', 'm2k', 'm3k', 'wlp', 'wl1k',
              'wl2k']
binsize = 16


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


gameboard_list.extend(add_boards())
gameareas_list.extend(add_areas())


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


def getType(slice):
    feature_vectors = createTileFeaturevectorArray()

    B = slice[:, :, 0].mean()
    G = slice[:, :, 1].mean()
    R = slice[:, :, 2].mean()

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

    distance = [48, 48, 48, 10000]
    type = ['None', 'None', 'None', 'Nej']

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

    sliceVector = calculateImageHistogramBinVector(slice, binsize)
    # distance = 300
    # type[3] = 'Nej'

    for tileType, featureVector in feature_vectors.items():
        new_distance = calculateEuclidianDistance(sliceVector, featureVector)
        #print(new_distance)
        if new_distance < distance[3]:
            distance[3] = new_distance
            type[3] = tileType

    return [type, distance]


def calculateEuclidianDistance(feature_vector1, feature_vector2):
    dist = np.sqrt(np.sum((feature_vector1 - feature_vector2) ** 2))
    return dist


def calculateImageHistogramBinVector(image, bins: int, name,factor: int = 255):
    # Hvis Grayscale / har kun en farvekanal
    if (len(image.shape) == 2):
        hist = np.histogram(image, bins, [0, 256])

    # Hvis billedet har farvekanaler/er BGR
    elif (len(image.shape) == 3):
        # Vi laver et histrogram til hver farvekanal
        B_hist, B_bins = np.histogram(image[:, :, 0], bins, [0, 256])
        G_hist, G_bins = np.histogram(image[:, :, 1], bins, [0, 256])
        R_hist, R_bins = np.histogram(image[:, :, 2], bins, [0, 256])

        fig = plt.figure()
        fig.suptitle(name, fontsize=15)

        width = 0.7*(B_bins[1]-B_bins[0])
        plt.subplot(1,3,1)
        plt.ylim([0, 10000])
        plt.title('Blue')
        plt.bar(B_bins[:-1],B_hist,width=width,color='blue', label = 'Blue')
        plt.subplot(1,3,2)
        plt.ylim([0, 10000])
        plt.title('Green')
        plt.bar(G_bins[:-1],G_hist,width=width,color='green', label = 'Green')
        plt.subplot(1,3,3)
        plt.ylim([0, 10000])
        plt.title('Red')
        plt.bar(R_bins[:-1],R_hist,width=width,color='red', label = 'Red')

        plt.show()

        # Vi fyrer alle histogrammer ind i røven ad hinanden i et ny array 'hist' for at få det som en feature vektor
        #hist = np.concatenate((B_hist[0], G_hist[0], R_hist[0]))

    # normaliserer histogrammet således at værdierne ligger mellem 0 og 1
    """
    hist = hist.astype(np.float64)
    if (hist.max() != 0 or None):
        hist /= int(hist.max())
        hist *= factor
    """

    #return hist


def saveTileVectors():
    def saveVector(fileName):
        img = cv.imread(f'King Domino dataset/Cropped_Tiles/{fileName}.jpg')
        np.save(f'King Domino dataset/Cropped_Tiles/{fileName}', calculateImageHistogramBinVector(img, binsize, fileName))

    for tile in tile_types:
        saveVector(tile)


def createTileFeaturevectorArray():
    featureVectors = {}
    for tile in tile_types:
        featureVectors[tile] = np.load(f'King Domino dataset/Cropped_Tiles/{tile}.npy')

    return featureVectors


############################################ Method calls


saveTileVectors()
"""
ROI = return_single_image(gameboard_list, 17)
# mROI = cv.medianBlur(ROI, 5)
slices = slice_roi(ROI)


for y, row in enumerate(slices):
    for x, slice in enumerate(row):
        sliceType, distance = getType(slice)

        x_coord = int((x * ROI.shape[1] / 5) + 5)
        y_coord = int((y * ROI.shape[0] / 5) + ROI.shape[0] / 20)

        cv.putText(ROI, f'{sliceType[0]}: {int(distance[0])}', (x_coord, y_coord + 00), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        cv.putText(ROI, f'{sliceType[1]}: {int(distance[1])}', (x_coord, y_coord + 15), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        cv.putText(ROI, f'{sliceType[2]}: {int(distance[2])}', (x_coord, y_coord + 30), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        cv.putText(ROI, f'{sliceType[3]}: {int(distance[3])}', (x_coord, y_coord + 45), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        # print(f'({y, x}). (BGR):{int(slices[y][x][:, :, 0].mean()), int(slices[y][x][:, :, 1].mean()), int(slices[y][x][:, :, 2].mean())}. (Center,Border): {defineCenterAndBorder(slice)}')
"""



#cv.imshow('Roi_with_contours', ROI)
#cv.imshow('Slice', slices[4][2])
#cv.waitKey(0)
