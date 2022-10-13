import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

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


ROI = return_single_image(gameboard_list, 12)

slices = slice_roi(ROI)


for y in range(len(slices)):
    for x in range(len(slices[y])):
        print(f'({y},{x})')
        print("B-gennemsnit "+ str(slices[y][x][:,:,0].mean()))
        print("G-gennemsnit "+ str(slices[y][x][:,:,1].mean()))
        print("R-gennemsnit "+ str(slices[y][x][:,:,2].mean())+"\n")


cv.imshow('Roi_with_contours', ROI)
#cv.imshow('Slice', slices[0][0])

cv.waitKey(0)


