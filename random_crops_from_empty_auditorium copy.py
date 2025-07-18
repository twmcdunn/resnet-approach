import cv2 as cv
import json
import csv, sys
from os import listdir
import random

SEAT_WIDTH = 99
SEAT_HEIGHT = 76

with open('seatDims.json','r') as f:
    seatDims = json.loads(f.read())
    SEAT_HEIGHT = seatDims["SEAT_HEIGHT"]
    SEAT_WIDTH = seatDims["SEAT_WIDTH"]

IMAGE_PATH =  'emptyImages'#'cam-test/just-cerimony'#'cam-test/edm'
images = listdir(IMAGE_PATH)
assert len(images) > 0, "Can't find images!"

numOfNegatives = 1000
numOfEdmanNegatives = 550#int(numOfNegatives / 2)

FIRST_INDEX = (numOfNegatives - numOfEdmanNegatives) + 1
for i in range(numOfEdmanNegatives):
    imgPath = IMAGE_PATH + '/' + random.choice(images)
    img = cv.imread(imgPath)
    height, width = img.shape[:2]
    y = random.randrange(0,height - SEAT_HEIGHT)
    x = random.randrange(0,width - SEAT_WIDTH)
    crop = img[y:y+SEAT_HEIGHT,x:x+SEAT_WIDTH]
    path = 'negativesFromEdman/' + str(i + FIRST_INDEX) + '.jpg'
    cv.imwrite(path,crop)
