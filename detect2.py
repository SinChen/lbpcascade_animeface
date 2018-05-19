import cv2 as cv
import sys
import os
import numpy as np

current_folder = os.path.dirname(os.path.abspath(__file__))
input_folder = current_folder + '/image/'
output_folder = current_folder + '/result/'
output_face_folder = current_folder + '/output_face/'

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
if not os.path.isdir(output_face_folder):
    os.mkdir(output_face_folder)
    
def detect(inputfolder, cascade_file = current_folder + '/lbpcascade_animeface.xml'):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)
    face_number = len([name for name in os.listdir(output_face_folder) \
        if os.path.isfile(os.path.join(output_face_folder, name))]) + 1
    for filename in os.listdir(inputfolder):
        filepath = input_folder + filename 
        cascade = cv.CascadeClassifier(cascade_file)
        image = cv.imread(filepath, cv.IMREAD_COLOR)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray,
                                        # detector options
                                        scaleFactor = 1.1,
                                        minNeighbors = 5,
                                        minSize = (24, 24))
        for (x, y, w, h) in faces:
            crop = image[y:y+h, x:x+w]
            crop_reshape = cv.resize(crop, (28,28), interpolation = cv.INTER_CUBIC)
            cv.imwrite(output_face_folder + "face_" + str(face_number) + ".jpg", crop_reshape)
            #cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #cv.imshow("crop face", crop)
            face_number+=1

        # if(len(faces) > 0):
        #     cv.imwrite(output_folder + "out_" + filename + ".jpg", image)
            
        cv.imshow("AnimeFaceDetect", image)    
        cv.waitKey(33)

detect(input_folder)