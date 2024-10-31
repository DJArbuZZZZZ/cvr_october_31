import cv2
cap = cv2.VideoCapture(0)
a = 100
while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    frame[:a, :a] = [0,0,0]
    frame[height - a:, width- a:] = [0, 0, 0]
    frame[ :a, width - a :] = [0, 0, 0]
    frame[height - a:, :a] = [0, 0, 0]
    frame[height // 2 - a//2: height//2 + a//2,  width // 2 - a//2: width//2 + a//2] = [0, 0, 0]
    cv2.imshow('camera', frame)
    key = cv2.waitKey(1000)
    if key == ord('f'):
        break
cv2.destroyAllWindows()
cap.release()