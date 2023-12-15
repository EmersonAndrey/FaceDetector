import cv2

image = cv2.imread("fotos/img1.jpg") # Para detectar na outra foto basta alterar o caminho

algoritmoCarregado = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

result = algoritmoCarregado.detectMultiScale(imageGray)

for (x, y, width, heigth) in result:
    cv2.rectangle(image, (x, y), (x + width, y + heigth), (0,255,0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "Face", (x, y - 10), font, 0.5, (0,255,0),1, cv2.LINE_AA)

cv2.imshow("Detecting faces", image)
cv2.waitKey()