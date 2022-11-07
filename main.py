import cv2 as cv
import numpy as np
import math

gameboard_list = []
gameareas_list = []


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


def slice_cutter(slice):
    center = slice[int(slice.shape[0]/8):int((slice.shape[0]/8)*7), int(slice.shape[1]/8):int((slice.shape[1]/8)*7)]
    centerMean = [int(center[:,:,0].mean()),int(center[:,:,1].mean()),int(center[:,:,2].mean())]

    top_border = slice[0:int(slice.shape[0]/8),:]
    bot_border = slice[int(slice.shape[0]-slice.shape[0]/8):slice.shape[0],:]
    left_border = slice[int(slice.shape[0]/8):int(slice.shape[0]-slice.shape[0]/8),0:int(slice.shape[1]/8)]
    right_border = slice[int(slice.shape[0]/8):int(slice.shape[0]-slice.shape[0]/8),int(slice.shape[1]-slice.shape[1]/8):slice.shape[1]]

    border_mean = [int((top_border[:,:,0].mean()+bot_border[:,:,0].mean()+left_border[:,:,0].mean()+right_border[:,:,0].mean())/4),
                   int((top_border[:,:,1].mean()+bot_border[:,:,1].mean()+left_border[:,:,1].mean()+right_border[:,:,1].mean())/4),
                   int((top_border[:,:,2].mean()+bot_border[:,:,2].mean()+left_border[:,:,2].mean()+right_border[:,:,2].mean())/4)]

    cut_slice = [centerMean, border_mean]

    return cut_slice


def getType(slice):
    B = slice[:, :, 0].mean()
    G = slice[:, :, 1].mean()
    R = slice[:, :, 2].mean()

    meanArray = [['wp', 150, 80, 14], ['w1k', 113, 80, 43], ['fp', 22, 60, 46],
                 ['f1k', 27, 62, 57], ['dp', 7, 155, 179], ['d1k', 19, 142, 169],
                 ['mp', 26, 51, 59], ['m1k', 27, 68, 77], ['m2k', 26, 51, 61],
                 ['m3k', 33, 62, 73], ['wlp', 60, 100, 111], ['wl1k', 54, 93, 104],
                 ['wl2k', 48, 89, 103], ['gp', 28, 146, 100], ['g1k', 25, 116, 103], ['g2k', 34, 126, 114]]

    distance = 25.0
    type = 'Undefined'

    for i, tile_type in enumerate(meanArray):
        new_distance = math.sqrt((B - tile_type[1]) ** 2 + (G - tile_type[2]) ** 2 + (R - tile_type[3]) ** 2)
        if new_distance < distance:
            distance = new_distance
            type = tile_type[0]

    return type


ROI = return_single_image(gameboard_list, 16)

slices = slice_roi(ROI)

"""
for y in range(len(slices)):
    for x in range(len(slices[y])):
        print(f'({y},{x})')
        print("B-gennemsnit "+ str(slices[y][x][:,:,0].mean()))
        print("G-gennemsnit "+ str(slices[y][x][:,:,1].mean()))
        print("R-gennemsnit "+ str(slices[y][x][:,:,2].mean())+"\n")
"""

for y, row in enumerate(slices):
    for x, slice in enumerate(row):
        sliceType = getType(slice)
        x_coord = int((x*ROI.shape[1]/5)+ROI.shape[1]/10)
        y_coord = int((y * ROI.shape[0]/5) + ROI.shape[0]/10)
        slice_text = f'{sliceType}'
        cv.putText(ROI, slice_text, (x_coord, y_coord), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        print(f'({y,x}). (BGR):{int(slices[y][x][:,:,0].mean()),int(slices[y][x][:,:,1].mean()),int(slices[y][x][:,:,2].mean())}. (Center,Border): {slice_cutter(slice)}')


cv.imshow('Roi_with_contours', ROI)
cv.imshow('Slice', slices[1][3])
cv.waitKey(0)
