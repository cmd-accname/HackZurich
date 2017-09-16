import cv2

cap = cv2.VideoCapture('output_d1.avi')  # 640 x 480
# cap.set(cv2.CAP_PROP_FPS, 30.0)  # 30 to 60 should be fine
t = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.IMREAD_UNCHANGED)
    # gaussian = cv2.GaussianBlur(gray, (7, 7), 0.5)

    # Display the resulting frame
    gray = frame
    cv2.imshow('frame', gray)
    print(t)
    # if (t==570):
    cv2.imwrite('D:/Data/zurich/gray' + str(t) + '.png',gray)
    t += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()