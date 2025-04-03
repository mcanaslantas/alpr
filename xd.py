import cv2
import imutils
import numpy as np
import easyocr

image_path = r'C:\Users\Mehmet\PycharmProjects\alpr\crop.mp4'
cap = cv2.VideoCapture(image_path)


mask_path = r'C:\Users\Mehmet\PycharmProjects\alpr\mask.png'
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

background_subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=False)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("End of video.")
        break

    resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    # Resize the frame to the desired width and height
    masked_frame = cv2.bitwise_and(frame, frame, mask=resized_mask)
    # Process the resized frame (add your processing steps here)
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

    # Display the processed frame
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Masked Frame', masked_frame)

    # Use easyocr to read text (example)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(edged)

    # Print recognized text (example)
    for tuple_element in result:
        label = tuple_element[-2]  # Extract label
        confidence_score = tuple_element[-1]  # Extract confidence score
        print(label)

    # Wait for 'q' key to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
