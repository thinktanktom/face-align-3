import cv2
import numpy as np
def detect_faces(detection_model, gray_image_array):

    faces = detection_model.detectMultiScale (gray_image_array, 1.1, 4)
    #to the right
    if(len(faces)==0):
        gray_image = cv2.rotate(gray_image_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        faces1 = detection_model.detectMultiScale (gray_image, 1.1, 4)
        if(len(faces1)==1):
            return gray_image
    #to the left
    if(len(faces)==0):
        gray_image2 = cv2.rotate(gray_image_array, cv2.ROTATE_90_CLOCKWISE)
        faces2 = detection_model.detectMultiScale (gray_image2, 1.1, 4)
        if(len(faces2)==1):
            return gray_image2
    #rotate the image 180 degrees
    if(len(faces)==0):
        gray_image3 = cv2.rotate(gray_image_array, cv2.ROTATE_180)
        faces3 = detection_model.detectMultiScale (gray_image3, 1.1, 4)
        if(len(faces3)==1):
            return gray_image3
    return gray_image_array
    
