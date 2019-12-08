import csv


# header = ["Actual Number Plate", "Predicted Number Plate", "result"]

# list1 = ['1','2','3']
# list2 = ['4','5','6']
# x = "50$"


# with open('OCR Accuracy Result.csv','w',newline='') as f:
#     writer = csv.writer(f,delimiter =',')
#     writer.writerow(header)
#     writer.writerows(zip(list1,list2))

# cleanText = ['2','0','C','1','2','3','4']

# if len(cleanText) == 7:
#     fix = []
#     for i, char in enumerate(cleanText[:3]):
#         if char == '0':
#             cleanText[i] = 'O'
#         if char == '2':
#             cleanText[i] = 'Z'
            

# print(cleanText)

strChar = ['A','B','C','D','3','3','3']

if len(strChar) > 5:
    for x in strChar:
        if not x.isdigit():
            print(x)
          