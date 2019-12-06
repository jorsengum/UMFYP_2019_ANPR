# OCR

import cv2
import numpy as np
import glob
import pytesseract
import os
import csv
import PIL
from PIL import Image, ImageChops
from matplotlib import pyplot as plt


def preprocess(plates, plateNamelist, preprocessedDir):
    print("Begin plate image proprocessing for OCR.....")
    preprocessed_plates =[]
    for i, img in enumerate(plates):

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # cv2.imshow('gray', gray)

        ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        # cv2.imshow('thresh', thresh)

        threshMean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)
        # cv2.imshow('threshMean', threshMean)

        threshGauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 27)
        
        ratio = 200.0 / threshGauss.shape[1]
        dim = (200, int(threshGauss.shape[0] * ratio))

        resizedCubic = cv2.resize(threshGauss, dim, interpolation=cv2.INTER_CUBIC)

        img_processed = threshGauss

        imgTrim = trimBlankBorder(img_processed)

        bordersize = 5
        img_processed = cv2.copyMakeBorder(img_processed, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                    borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
       
        img, cntrs, _ = cv2.findContours(img_processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        h_margin_cntrs = h_filter(cntrs,15,0.1)          # filter height margin, remove any box that is higher than average

        sorted_cntrs, sorted_box = sortingBox(h_margin_cntrs)
        
        imgCrop = trimBox(sorted_box,7,2, img_processed) # list of box, padding, img to crop

        img_processed = imgCrop

        bordersize = 5
        img_processed = cv2.copyMakeBorder(img_processed, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                    borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        cv2.imwrite(preprocessedDir + plateNamelist[i] + ".jpg", img_processed)
        
        preprocessed_plates.append(img_processed)
        
    return preprocessed_plates


def OCR(preprocessed):
    print("Begin OCR.....")
    OCROutput = []

    for i, image in enumerate(preprocessed):
        # image = cv2.imread(image)

        # OCR
        config = '-l eng --oem 1 --psm 3'
        text = pytesseract.image_to_string(image, config=config)

        validChars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        cleanText = []

        for char in text:
            if char in validChars:
                cleanText.append(char)

        plate = ''.join(cleanText)

        OCROutput.append(plate)

    with open('OCROutput.csv','w',newline='') as f:
        writer = csv.writer(f)
        for ocr in OCROutput:
            writer.writerow([ocr])

    return OCROutput


def CheckOCR(OCROutput, listOfNumberPlates):

    OCRComparison = [listOfNumberPlates,OCROutput]

    OCRAccuracy = sum(x in OCROutput for x in listOfNumberPlates)/(len(listOfNumberPlates)-1)*100 # minus 1 because there is an empty item in the list for some reason.
    OCRAccuracy = round(OCRAccuracy,2)

    print("Actual Plate Number vs Predicted Plate Number\n")

    for x in zip(*OCRComparison):
        print (*x)
    
    print("OCR Accuracy = {} %".format(OCRAccuracy))

    return OCRAccuracy


def trimBlankBorder(im):
    im = Image.fromarray(im)

    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        im = im.crop(bbox)

    return np.asarray(im) # convert pil_img to cv_img


def h_filter(contours,margin,ratio):
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    aspect_ratio_cntrs = []
    count = 0
    sum_h = 0

    for c in contours[:10]:
        x,y,w,h = cv2.boundingRect(c)
        aspect_ratio = h/w

        if aspect_ratio < ratio:
            continue            

        count+=1
        sum_h = sum_h+h
        aspect_ratio_cntrs.append(c)

    avg_h = sum_h/count
    
    filtered_height_contours = []
    for c in aspect_ratio_cntrs[:10]:
        x,y,w,h = cv2.boundingRect(c)
        h_margin = abs((h - avg_h)/avg_h*100)

        if h > avg_h and h_margin > margin:
            continue    

        filtered_height_contours.append(c)

    return filtered_height_contours


def drawbox(contours,img):
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    return

def sortingBox(contours):
    # Sort contours left to right
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key = lambda b:b[1][0], reverse = False))
	# return the list of sorted contours and bounding boxes
    return contours, boundingBoxes

def trimBox(sorted_box, x_padding, y_padding, img):
    # Trims ROI based on all the bounding box limits
    x_coor  = []
    y_coor  = []
    y_height= []
        
    for b in sorted_box:
        x_coor.append(b[0])
        y_coor.append(b[1])
        y_height.append(b[3])
    
    x_end_index = x_coor.index(max(x_coor))
    y_max_index = y_coor.index(max(y_coor))
    y_min_index = y_coor.index(min(y_coor))
    y_max_height_index = y_height.index(max(y_height))
    y_min_max_dist = abs(sorted_box[y_max_index][1] - sorted_box[y_min_index][1])

    x_start = max((min(x_coor) - x_padding),0)
    x_end   = sorted_box[x_end_index][0] + sorted_box[x_end_index][2] + x_padding

    y_start = max((min(y_coor) - y_padding),0)
    y_end   = sorted_box[y_max_index][1] + sorted_box[y_max_height_index][3] + y_min_max_dist + y_padding

    cropped_image = img[y_start: y_end, x_start:x_end]

    return cropped_image


# def WriteNumberPlateList(NumberPlateList):
#     with open('listOfNumberPlates.csv','w',newline='') as f:
#         writer = csv.writer(f)
#         for file in NumberPlateList:
#             writer.writerow([file])

#     return

