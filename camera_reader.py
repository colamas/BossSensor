# -*- coding:utf-8 -*-
import cv2

from boss_train import Model
from image_show import show_image


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    model = Model()
    model.load()

    while True:
        _, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
        if len(facerect) > 0:
            print('face detected')
            for rect in facerect:
                x, y = rect[0:2]
                width, height = rect[2:4]
                image = frame[y - 10: y + height, x: x + width]
                result = model.predict(image)
                if result == 0:
                    print('Boss is approaching')
                    show_image()
                else:
                    print('Not boss')

        k = cv2.waitKey(100)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
