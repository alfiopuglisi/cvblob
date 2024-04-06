
import cv2
import numpy as np
from collections import namedtuple
from guietta import Gui, M, HValueSlider, ___, connect

filename = '13PE12024hGArea1Cel25.jpg'

Param = namedtuple('Param', 'name range')
Range = namedtuple('Range', 'start stop step')

myparams = [
        Param('filterByColor', range(1)),
        Param('blobColor', range(255)),
        Param('filterByArea', range(1)),
        Param('minArea', range(10000)),
        Param('maxArea', range(10000)),
        Param('filterByCircularity', range(1)),
        Param('minCircularity', Range(0, 1, 0.01)),
        Param('maxCircularity', Range(0, 1, 0.01)),
        Param('filterByConvexity', range(1)),
        Param('minConvexity', Range(0, 1, 0.01)),
        Param('maxConvexity', Range(0, 1, 0.01)),
        Param('filterByInertia', range(1)),
        Param('minInertiaRatio', Range(0, 1, 0.01)),
        Param('maxInertiaRatio', Range(0, 1, 0.01)),
        Param('minThreshold', range(255)),
        Param('maxThreshold', range(255)),
        Param('thresholdStep', range(255)),
        Param('minDistBetweenBlobs', range(100)),
        Param('minRepeatability', range(100)),
        ]

rows = [[f'_{p.name}', HValueSlider(p.name, p.range), ___] for p in myparams]

gui = Gui([M('image')],
          *rows
          )

img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
gui.image = img


def refresh(gui, *args):
    params = cv2.SimpleBlobDetector_Params()
    for p in myparams:
        setattr(params, p.name, getattr(gui, p.name))

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    imk = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_gray = cv2.cvtColor(imk, cv2.COLOR_BGR2GRAY)
    gui.image = img_gray


# Default params
params = cv2.SimpleBlobDetector_Params()
for p in myparams:
    value = getattr(params, p.name)
    if type(value) is bool:
        value = 1 if value else 0
    if value > 2**31-1:
        value= 2**31-1
    print(p.name, value)
    setattr(gui, p.name, value)
    connect(gui.widgets[p.name], slot=refresh)

gui.run()
