import os

from face import *
from PIL import Image
import matplotlib.pyplot as plt


def save_faces(len, image, boxes, save_path='images'):
    image = np.array(image)
    for i in range(len):
        box = boxes[0]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        face = image[y1: y2, x1: x2, :]
        face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        path = save_path + "face_" + str(i) + ".jpg"
        print('[save] ', path)
        cv2.imwrite(path, face)

fs = FaceSystem()
# 预测人脸的例子
for name in os.listdir('./images/train/1'):
    img = Image.open('./images/train/1/'+name)
    result = fs.face_detect(img)
    fs.show_face_boxes(img, result)

# 将人脸切割保存
    #save_faces(len(os.listdir('./images/train/1')), img, result, './images/train/1/')







