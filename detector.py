import os
from PIL import Image
from face import FaceSystem

fs = FaceSystem()
face1 = fs.video_face_one()
feature1 = fs.get_face_feature(face1)

registered_names = os.listdir('images/train/')
st = False
for name in registered_names:
    if st:
        break
    for i in os.listdir('images/train/'+name):
        path = 'images/train/'+name+'/'+i
        face2 = Image.open(path)
        feature2 = fs.get_face_feature(face2)
        dist = fs.feature_compare(feature1, feature2)
        print("name:{}, dist:{}".format(name, dist))
        if dist < 0.5:
            st = True
            print('{} Check-in successful'.format(name))
            break
