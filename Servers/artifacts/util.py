import cv2
import json
import numpy as np
import base64
import joblib
from wavelet import w2d

__class_name_to_number={}
__class_number_to_name={}
__model=None

def classify_image(image_base64_data, file_path=None):
    images=get_cropped_image_2(file_path,image_base64_data)
    result =[]
    for img in images:
        scaled_rawImage=cv2.resize(img,(32,32))
        image_haar=w2d(img,'db1',5)
        scaled_image_haar=cv2.resize(image_haar, (32,32))
        #vertical stacking using numpy functions
        combined_img=np.vstack((scaled_rawImage.reshape(32*32*3,1), scaled_image_haar.reshape(32*32,1)))
        len_image_array= (32*32*3)+(32*32)
        final=combined_img.reshape(1, len_image_array).astype(float)

        result.append(__model.predict(final)[0])
    return result
def load_artifacts():
    print("loading saved artifacts....start")
    global __class_name_to_number
    global __class_number_to_name
    with open("./artifacts/Class.Json", "r") as f:
        __class_name_to_number=json.load(f)
        __class_number_to_name={v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open("./artifacts/saved_model.pkl","rb") as f:
            __model = joblib.load(f)
    print("loading save artifacts...done")

def get_cv2_image_from_base64_string(bs64str):
    encoded_data=bs64str.split(',')[1]
    nparr=np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img=cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    return img

def  get_cropped_image_2(image_path, image_base64_data):
    face_cascade=cv2.CascadeClassifier(".\OpenCv\haarcascade_frontalface_default.xml")
    eye_cascade=cv2.CascadeClassifier(".\OpenCv\haarcascade_eye.xml")
    if image_path:
        img=cv2.imread(image_path)
    else:
        img=get_cv2_image_from_base64_string(image_base64_data)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    cropped_faces=[]
    for (x,y,w,h) in faces:
           roi_gray= gray[y:y+h, x:x+w]
           roi_color=img[y:y+h, x:x+w]
           eyes=eye_cascade.detectMultiScale(roi_gray)
           if len(eyes) >=2:
                 cropped_faces.append(roi_color)
    return cropped_faces



def get_bs64_test_image_for_virat():
    with open('viratkohlibs64.txt') as f:
        return f.read()

if __name__ == "__main__" :
    load_artifacts()
    print(classify_image(get_bs64_test_image_for_virat(), None))
