import cv2

loadModel = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

webcam = cv2.VideoCapture(0)

while True:
    status, frame = webcam.read()

    imageGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = loadModel.detectMultiScale(imageGray)

    for (x, y, width, heigth) in result:
        cv2.rectangle(frame, (x, y), (x + width, y + heigth), (0,255,0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Face", (x, y - 10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("Detecting faces", frame)

    if not status or cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()