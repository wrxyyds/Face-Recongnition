from face import FaceSystem
import train
import os

fs = FaceSystem()
name = input("Enter your name: ")
img_path = 'images/train/' + name + '/'
fs.video_face_save(save_path=img_path)
if os.path.exists('face/facenet/weights/transferred_facenet_model.pt'):
    train.train('./images/train/', './face/facenet/weights/transferred_facenet_model.pt', './face/facenet/weights'
                                                                                          '/transferred_facenet_model'
                                                                                          '.pt')
else:
    train.train('./images/train/', './face/facenet/weights/model_resnet34_triplet.pt', './face/facenet/weights'
                                                                                       '/transferred_facenet_model.pt')
