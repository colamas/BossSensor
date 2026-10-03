# -*- coding:utf-8 -*-
import cv2

from boss_train import Model
from image_show import show_image


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    # カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise IOError('Failed to load cascade classifier: ' + cascade_path)

    model = Model()
    model.load()
    while True:
        ok, frame = cap.read()
        if not ok:
            print('Failed to read frame from camera')
            break

        # グレースケール変換
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 物体認識（顔認識）の実行
        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
        if len(facerect) > 0:
            print('face detected')
            for rect in facerect:
                x, y = rect[0:2]
                width, height = rect[2:4]
                # 顔の少し上まで含める（画像の端を越えないようにする）
                image = frame[max(y - 10, 0): y + height, x: x + width]

                result = model.predict(image)
                if result == 0:  # boss
                    print('Boss is approaching')
                    show_image()
                    break
                else:
                    print('Not boss')

        #100msecキー入力待ち
        k = cv2.waitKey(100)
        #Escキーを押されたら終了
        if k == 27:
            break

    #キャプチャを終了
    cap.release()
    cv2.destroyAllWindows()
