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

IMAGE_PATH =  'resnet-approach/emptyImages'#'cam-test/just-cerimony'#'cam-test/edm'
images = listdir(IMAGE_PATH)
assert len(images) > 0, "Can't find images!"

FIRST_INDEX = 100
for i in range(5000):
    img = random.choice(images)
    height, width = img.shape[:2]
    y = random.randrange(0,height - SEAT_HEIGHT)
    x = random.randrange(0,width - SEAT_WIDTH)
    crop = img[y:y+SEAT_HEIGHT,x:x+SEAT_WIDTH]
    path = 'negativesFromEdman/' + str(i + FIRST_INDEX) + '.jpg'
    cv.imwrite(path,crop)
