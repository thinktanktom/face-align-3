import cv2
import face_detect
import dlib
import euclidean_distance
import numpy as np
import math
from PIL import Image
import time
import threading
import queue

start = time.time()
que = queue.Queue()
a = queue.Queue()
b = queue.Queue()
c = queue.Queue()
threads_list = list()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
image = cv2.imread('images/real_1.jpg')
img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#Thread 1
t1 = threading.Thread(target=lambda q, arg1,arg2: q.put(face_detect.detect_faces(arg1,arg2)), args=(que,face_cascade, img_gray))
t1.start()
t1.join()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
align1 = que.get()
detections = detector(align1,1)
faces = dlib.full_object_detections()
for det in detections:
    faces.append(predictor(align1, det))
right_eye = [[face.part(i) for i in range(36, 42)] for face in faces]
right_eye = [[(i.x, i.y) for i in eye] for eye in right_eye]          # Convert out of dlib format
right_eye = right_eye[0][0]
right_eye_x = right_eye[0]
right_eye_y = right_eye[1]
left_eye = [[face.part(i) for i in range(42, 48)] for face in faces]
left_eye = [[(i.x, i.y) for i in eye] for eye in left_eye]
left_eye = left_eye[0][0]
left_eye_x = left_eye[0]
left_eye_y = left_eye[1]
if left_eye_y < right_eye_y:
   point_3rd = (right_eye_x, left_eye_y)
   direction = -1 #rotate same direction to clock
   print("rotate to clock direction")
else:
   point_3rd = (left_eye_x, right_eye_y)
   direction = 1 #rotate inverse direction of clock
   print("rotate to inverse clock direction")
   print("rotate to inverse clock direction")
#Thread 2,3,4
t2 = threading.Thread(target = lambda e, arg1,arg2: e.put(euclidean_distance.euclidean_distance(arg1,arg2)), args = (a, left_eye,point_3rd))
t3 = threading.Thread(target = lambda e, arg1,arg2: e.put(euclidean_distance.euclidean_distance(arg1,arg2)), args = (b, right_eye,left_eye))
t4 = threading.Thread(target = lambda e, arg1,arg2: e.put(euclidean_distance.euclidean_distance(arg1,arg2)), args = (c, right_eye,point_3rd))
t2.start()
threads_list.append(t2)
t3.start()
threads_list.append(t3)
t4.start()
threads_list.append(t4)
for t in threads_list:
   t.join()
a = a.get()
b = b.get()
c = c.get()
cos_a = (b*b + c*c - a*a)/(2*b*c)
print("cos(a) = ", cos_a)
 
angle = np.arccos(cos_a)
print("angle: ", angle," in radian")
 
angle = (angle * 180) / math.pi
print("angle: ", angle," in degree")

if direction == -1:
   angle = 90 - angle
new_img = Image.fromarray(align1)
new_img = np.array(new_img.rotate(direction * angle))
faces2 = face_cascade.detectMultiScale(new_img, 1.3, 5)
face_x, face_y, face_w, face_h = faces2[0]
new_img = new_img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
cv2.imshow('aligned',new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
end = time.time()
print(f"Runtime of the program is {end - start}")