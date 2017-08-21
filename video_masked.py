
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.GaussianBlur(frame, (5,5), 0)
    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([150,50,50])
    upper_blue = np.array([200,255,255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(hsv,hsv, mask= mask)


    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('masked', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()