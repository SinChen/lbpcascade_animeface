import cv2 as cv
import sys
import os.path

def detect(filename, cascade_file = "./lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv.CascadeClassifier(cascade_file)
    image = cv.imread(filename, cv.IMREAD_COLOR)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv.imshow("AnimeFaceDetect", image)
    cv.waitKey(0)
    cv.imwrite("out.png", image)


if len(sys.argv) != 2:
    sys.stderr.write("usage: detect.py <filename>\n")
    sys.exit(-1)
detect(sys.argv[1])
