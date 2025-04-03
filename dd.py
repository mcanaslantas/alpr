import cv2
import imutils
import numpy as np
import easyocr
import threading
import time
import torch
print(torch.cuda.get_device_name(0))

class ProcessingThread(threading.Thread):
    def __init__(self, image):
        threading.Thread.__init__(self)
        self.image_path = image

    def run(self):
        img = cv2.imread(self.image_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

        edged = cv2.Canny(bfilter, 30, 200)

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break


        mask = np.zeros(gray.shape, np.uint8)
        img = cv2.drawContours(mask, [location], 0, 255, -1)
        img = cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped = gray[x1:x2 + 1, y1:y2 + 1]

        reader = easyocr.Reader(['en'], gpu=True)
        result = reader.readtext(cropped)
        print(location)
        for tuple_element in result:
            label = tuple_element[-2]
            print(label, end="")

        print()
        return result


if __name__ == '__main__':

    # Path to your video file
    path = r'C:\Users\Mehmet\PycharmProjects\alpr\crop.mp4'
    cap = cv2.VideoCapture(path)
    start_time = time.time()

    desired_width = 1280
    desired_height = 720

    while True:
        ret, frame = cap.read()
        resized_frame = cv2.resize(frame, (desired_width, desired_height))

        if not ret:
            print("End of video.")
            break

        cv2.imshow('Original Frame', resized_frame)
        current_time = time.time()

        if current_time - start_time >= 3:

            cv2.imwrite('ss.jpg', resized_frame)
            start_time = current_time
            processing_thread = ProcessingThread('ss.jpg')
            processing_thread.start()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()