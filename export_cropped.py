import cv2 as cv
import json
import csv, sys
from os import listdir

DATA_SET_ROOT = 'trials/1_reolink1_wheaton_academy'#'trials/1_reolink1_wheaton_academy'

SEAT_WIDTH = 99
SEAT_HEIGHT = 76

with open('seatDims.json','r') as f:
    seatDims = json.loads(f.read())
    SEAT_HEIGHT = seatDims["SEAT_HEIGHT"]
    SEAT_WIDTH = seatDims["SEAT_WIDTH"]

seat_coords = []
seat_numbers = []
with open(DATA_SET_ROOT + '/seats.csv', newline='') as seats_csv:
    csvreader = csv.reader(seats_csv)
    for row in csvreader:
        if len(row) < 3:
            sys.exit(f'Parse error on line {csvreader.line_num}.')
        seat_coords.append((int(row[0]),int(row[1])))
        seat_numbers.append(row[2])

IMAGE_PATH =  DATA_SET_ROOT + '/images'#'cam-test/just-cerimony'#'cam-test/edm'
images = listdir(IMAGE_PATH)
assert len(images) > 0, "Can't find images!"

targetsNums = []
with open(DATA_SET_ROOT + '/targets.csv', encoding='utf-8', newline='') as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        targetsNums.append(row[2])

index = 0
for i in range(len(seat_coords)):
    seatx, seaty = seat_coords[i]
    for image_index in range(1):#range(0,len(images), int(len(images) / 2)):
        #print("IMAGE InDEX " + str(image_index))
        if seat_numbers[i] in targetsNums:
            classification = 'non_person'
            break
        chapel_picture = cv.imread(f"{IMAGE_PATH}/{images[image_index]}",cv.IMREAD_COLOR)
        
        seat_cropped = chapel_picture[seaty:seaty+SEAT_HEIGHT, seatx:seatx+SEAT_WIDTH]
        
        classification = "person"
       
        path = 'personsa/a' + str(index) + '.jpg'
        cv.imwrite(path,seat_cropped)
        index += 1
    #print(f"PROGRESS{100 * i / len(seat_coords)} %")

