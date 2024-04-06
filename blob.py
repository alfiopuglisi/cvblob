
import cv2
import numpy as np
from guietta import Gui, M, HValueSlider, ___

filename = '13PE12024hGArea1Cel25.jpg'

gui = Gui([M('image')],
          ['_minDistBetweenBlobs', HValueSlider('minDistBetweenBlobs', range(100)), ___],
          ['_filterByCircularity', HValueSlider('filterByCircularity', range(1)),   ___],
          ['_minThreshold', HValueSlider('minThreshold', range(255)), ___],
          ['_maxThreshold', HValueSlider('maxThreshold', range(255)), ___],
          )

img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
gui.image = img

# Default params
params = cv2.SimpleBlobDetector_Params()
gui.minDistBetweenBlobs = params.minDistBetweenBlobs
gui.filterByCircularity = 1 if params.filterByCircularity else 0
gui.minThreshold = params.minThreshold
gui.maxThreshold = params.maxThreshold


def refresh(gui):
    params = cv2.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = gui.minDistBetweenBlobs
    params.filterByCircularity = bool(gui.filterByCircularity)
    params.minThreshold = gui.minThreshold
    params.maxThreshold = gui.maxThreshold

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    imk = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_gray = cv2.cvtColor(imk, cv2.COLOR_BGR2GRAY)
    gui.image = img_gray


with gui.minDistBetweenBlobs, gui.filterByCircularity, gui.minThreshold, gui.maxThreshold:
     refresh(gui)

gui.run()
