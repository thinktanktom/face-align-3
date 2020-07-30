# Facial Alignment part 3

This program combines attributes from the first two facial alignment repositories to align the image perfectly without much overhead.

# Working

It is a well-known fact that the python library dlib is slightly better than cv2.CascadeClassifier albeit slower. So to recognise the face in the first function(face_detect.py), I've used dlib whereas to detect the right and left eyes I used the cascade classifier. I've also implemented threading to somewhat reduce runtime.