from .mtcnn import Detector
from .mtcnn import draw_bboxes
from .mtcnn import get_max_boxes
from .utils import *
from .facenet import FaceExtractor
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image


class FaceSystem:
    def __init__(self):
        self.face_detector = Detector()
        self.face_extractor = FaceExtractor()

    def face_detect(self, image):
        """
        predict the locations of faces in the image
        """
        boxes, landmarks = self.face_detector.detect_faces(image)
        return boxes

    def save_faces(self, image, boxes, save_path='images'):
        image = np.array(image)
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            face = image[y1: y2, x1: x2, :]
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            path = os.path.join(save_path, "face_" + str(i) + ".jpg")
            print('[save] ', path)
            cv2.imwrite(path, face)

    def show_face_boxes(self, image, boxes):
        """
        draw face boxes on the image
        """
        result = draw_bboxes(image, boxes)
        show_image(result)

    def video_face_reg(self, cam_id=0):
        cap = cv2.VideoCapture(cam_id)
        while True:
            ret, image = cap.read()
            image = Image.fromarray(image, mode='RGB')
            faces = self.face_detect(image)
            image = draw_bboxes(image, faces)
            image = np.array(image)
            image = image.astype(np.uint8)
            cv2.imshow("face", image)
            cv2.waitKey(1)

    def get_face_feature(self, face):
        feature = self.face_extractor.extractor(face)
        return feature

    def feature_compare(self, feature1, feature2):
        dist = np.sqrt(np.sum(np.square(np.abs(feature1 - feature2))))
        return dist

    def video_face_save(self, cam_id=0, save_path='images/train/'):
        # 创建保存文件夹
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cap = cv2.VideoCapture(cam_id)
        face_count = 0  # 初始化人脸计数器

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(frame, mode='RGB')
            faces = self.face_detect(image)

            for i, box in enumerate(faces):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                face = np.array(image)[y1:y2, x1:x2]
                face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

                # 保存人脸图片
                face_path = save_path + f"face_{face_count}.jpg"
                cv2.imwrite(face_path, face)
                face_count += 1
                print(f"[save] {face_path}")

                if face_count >= 2:
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def video_face_one(self, cam_id=0):
        cap = cv2.VideoCapture(cam_id)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(frame, mode='RGB')
            faces = self.face_detect(image)

            if len(faces) > 0:
                # 选择第一个检测到的人脸
                box = faces[0]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                face = np.array(image)[y1:y2, x1:x2]
                face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

                cap.release()
                cv2.destroyAllWindows()
                return face

            cv2.imshow("face", np.array(image).astype(np.uint8))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return None
