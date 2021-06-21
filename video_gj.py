import tensorflow as tf
from PIL import Image
import os
import cv2
import numpy as np
#import imutils
from yolo import YOLO
import time

yolo = YOLO()
cap = cv2.VideoCapture(0)
fps = 0.0

#mode = 'full'
mode = 'roi'

if mode == 'full':
    while True:
        t1 = time.time()

        ret, frame = cap.read() #读取某一帧
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 格式转变，BGRtoRGB
        frame = Image.fromarray(np.uint8(frame)) # 转变成Image
        r_image = yolo.detect_image(frame)
        img = np.asarray(r_image)
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # RGBtoBGR

        # 显示fps
        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", frame)

        #r_image.show()

        if cv2.waitKey(1) & 0xff == 27:
            break

elif mode == 'roi':
    while True:
        t1 = time.time()

        ret, frame = cap.read()  # 读取某一帧
        r_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 格式转变，BGRtoRGB
        r_image = Image.fromarray(np.uint8(r_image))  # 转变成Image

        # 检测并输出
        r_image = yolo.detect_image(r_image)
        r_image = np.asarray(r_image)
        r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)  # RGBtoBGR

        # 显示fps
        fps = (fps + (1. / (time.time() - t1))) / 2
        #print("fps= %.2f" % (fps))
        r_image = cv2.putText(r_image, "WHUT fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # 绘制危险区域框并展示
        # (x1,y1) (x2,y2)
        cv2.rectangle(r_image, (50, 50), (400, 450), (0, 255, 0), 2)
        cv2.imshow("video", r_image)

        if cv2.waitKey(1) & 0xff == 27:
            break

cap.release()
cv2.destroyAllWindows()
print('end')

