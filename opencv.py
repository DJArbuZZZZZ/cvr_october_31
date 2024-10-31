import cv2
from pprint import pprint
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    print(ret)
    for el in frame:
        for i in el:
            print(i, end= " ")
        print()
    cv2.imshow('camera', frame)
    cv2.waitKey(1)
    break