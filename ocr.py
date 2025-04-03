import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import re

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image_path = r'C:\Users\Mehmet\PycharmProjects\alpr\2.jpg'
img = cv2.imread(image_path)
cv2.imshow('BLACK', img)


def select_roi(image):
    roi = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=False)
    cv2.destroyWindow("Select ROI")  # Close the window after selection
    return roi


selected = select_roi(img)
print(type(selected))
selected = img[int(selected[1]):int(selected[1] + selected[3]),
           int(selected[0]):int(selected[0] + selected[2])]
print(type(selected))
gray = cv2.cvtColor(selected, cv2.COLOR_BGR2GRAY)
cv2.imshow('grayt', gray)
blurred = cv2.GaussianBlur(gray, (9, 9), 0)
cv2.imshow('blur', blurred)
_, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]


hIm, wIm = selected.shape[:2]

boxes = pytesseract.image_to_string(threshold, lang='eng',
                                    config=r'--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOUPRSTYZ')
cv2.imshow('threshold', threshold)

d = pytesseract.image_to_data(img, output_type=Output.DICT)
keys = list(d.keys())
plates = r'^([0-7][0-9]|8[0-1])[A-Z]\d{2,4}$'
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        if re.match(plates, d['text'][i]):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

for b in boxes.splitlines():
    print(b)
    b = b.split(' ')
    print(b)
cv2.waitKey(0)
