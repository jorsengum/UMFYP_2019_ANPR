# Main.py

import cv2
import numpy as np
import os

import DetectPlates 
import DetectChars 
import OCR
import glob
import csv

# File system paths ###########################################################################################################################################

# Pre-defined training set for KNN Model on character classification, used for more precise character detection only.
CLASS_TXT_DIR = "C:\\Users\\jorseng\\OneDrive - 365.um.edu.my\\UM\\Units\\01 WQD7002 Research Project\\FYP\\ANPRCode\\github\\UMFYP_2019_ANPR\\classifications.txt"
FLAT_IMG_DIR = "C:\\Users\\jorseng\\OneDrive - 365.um.edu.my\\UM\\Units\\01 WQD7002 Research Project\\FYP\\ANPRCode\\github\\UMFYP_2019_ANPR\\flattened_images.txt"

# Dataset path
DATASET_DIR = "C:\\Users\\jorseng\\OneDrive - 365.um.edu.my\\UM\\Units\\01 WQD7002 Research Project\\FYP\\ANPRCode\\github\\UMFYP_2019_ANPR\\Dataset\\"

# Output path for extracted plates
EXTRACTED_PLATES_DIR = DATASET_DIR+ "extractedPlates\\"

# Output path for preprocessed plates for OCR
PREPROCESSED_PLATES_DIR = EXTRACTED_PLATES_DIR + "preprocessed\\"

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

# To show image or not.
showstep_plate = False
showstep_ocr = False

def main():
    print("Loading Images ...")
    # KNN ML Model ################################################################################################################
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN(CLASS_TXT_DIR, FLAT_IMG_DIR)         # attempt KNN training

    if blnKNNTrainingSuccessful == False:                               
        print("\nerror: KNN traning was not successful\n")              
        return                                                          
    # end if

    # Loading Images ##############################################################################################################
    BR_IMGS = os.listdir(DATASET_DIR)
    BR_IMGS = [img for img in BR_IMGS if ".jpg" in img]
    extractedPlates = []
    numberPlateList = []
    filename = []

    # Write extracted plates images to path
    print("Begin Pre-processing images ...")
    for BR_IMG in BR_IMGS:
        BR_IMG_path = DATASET_DIR + BR_IMG
        imgOriginal = cv2.imread(BR_IMG_path)
        listOfPossiblePlates = DetectPlates.detectPlates(imgOriginal)
        listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

        # Sorting list of possible plates in DESC order (the one with most chars recognized)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)
        licPlate = listOfPossiblePlates[0]
        extractedPlates.append(licPlate.imgPlate)
        numberPlateList.append(BR_IMG.rsplit('.',1)[0])

        cv2.imwrite(EXTRACTED_PLATES_DIR+BR_IMG,licPlate.imgPlate)

    # Export list of number plate to csv
    WriteNumberPlateList(numberPlateList)
    print("Completed pre-processing and export list of number plates.")

    # Pre-process extracted plates and export processed image file
    print("Begin Image Pre-processing for OCR..")
    ocr_preprocessed_plates = OCR.preprocess(extractedPlates,numberPlateList,PREPROCESSED_PLATES_DIR)

    # OCR the preprocessed extracted plates and export the OCR Output to csv
    OCROutput = OCR.OCR(ocr_preprocessed_plates)
    print("Completed Image Pre-processing for OCR and export to csv.")

    # Accuracy result based on predicted number plate vs actual number plates
    OCR_Result = OCR.CheckOCR(OCROutput,numberPlateList)
    
    # Export Result to csv
    ExportActualAndPredictedNumberPlate(numberPlateList,OCROutput)
    
    return # end of main()

###################################################################################################
# Export list of number plate
def WriteNumberPlateList(NumberPlateList):
    with open('listOfNumberPlates.csv','w',newline='') as f:
        writer = csv.writer(f)
        for file in NumberPlateList:
            writer.writerow([file])
    return

def ExportActualAndPredictedNumberPlate(actualNumberPlate,predictedNumberPlate):
    header = ["Actual_Number_Plate", "Predicted_Number_Plate"]
    with open('OCR Accuracy Result.csv','w',newline='') as f:
        writer = csv.writer(f,delimiter =',')
        writer.writerow(header)
        writer.writerows(zip(actualNumberPlate,predictedNumberPlate))
    return

#================================================================================
if __name__ == "__main__":
    main()

