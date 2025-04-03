import cv2
import imutils
import numpy as np
import easyocr


image_path = r'C:\Users\Mehmet\PycharmProjects\alpr\123.jpg'
img = cv2.imread(image_path)
cv2.imshow('BLACK', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('GRAY', gray)
bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv2.Canny(bfilter, 30, 200) #Edge detection
cv2.imshow('edged', edged)


keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]


location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

print(location)

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]
cv2.imshow('cropped', cropped_image)


reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)

for tuple_element in result:
    label = tuple_element[-2]  # Extract label
    confidence_score = tuple_element[-1]  # Extract confidence score

    # Print the label without a newline
    print(label, end="")

# Print the newline
print()


cv2.waitKey(0)
