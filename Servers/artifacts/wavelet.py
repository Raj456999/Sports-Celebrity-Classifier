import numpy as np
import cv2
import pywt
def w2d(image, mode='haar', level=1):
    image_array=image
    image_array=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_array=np.float32(image_array)
    image_array/= 255
    coeffecients=pywt.wavedec2(image_array,mode,level=level)
    coeffecients_H=list(coeffecients)
    coeffecients_H[0]*=255
    image_array_H=pywt.waverec2(coeffecients_H,mode)
    image_array_H*=255
    image_array_H=np.uint8(image_array_H)
    return image_array_H



