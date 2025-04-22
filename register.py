from face import FaceSystem
import train

fs = FaceSystem()
name = input("Enter your name: ")
img_path = 'images/train/' + name + '/'
fs.video_face_save(save_path=img_path)
train.train()

