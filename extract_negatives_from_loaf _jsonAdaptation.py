from os import listdir
from PIL import Image
import random

import json

jsonData = None

with open('loaf_data/annotations/annotations/resolution_2k/instances_train.json','r') as f:
    jsonData = json.loads(f.read())

print(str(jsonData))

dirArr = ['/train','/val','/test']
def do_boxes_overlap(box1, box2):
    """
    Check if two bounding boxes overlap.
    
    Args:
        box1: Tuple (x1, y1, x2, y2) for the first box.
        box2: Tuple (x1, y1, x2, y2) for the second box.
    
    Returns:
        True if the boxes overlap, False otherwise.
    """
    # Unpack the coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Check for overlap
    if x1_max < x2_min or x2_max < x1_min:
        return False  # No horizontal overlap
    if y1_max < y2_min or y2_max < y1_min:
        return False  # No vertical overlap

    return True  # Boxes overlap


negCount = 0

imgFolder = 'loaf_data/resolution_2k/train'#'loaf_data/images'

loafImgs = listdir(imgFolder)

personCount = 0
outDir = 'loaf_data/loaf_negatives'


for i in range(len(loafImgs)):
    if i % 10000 == 0:
        print(str(i/len(loafImgs)) + ' imgs in negatives')
    imgFile = loafImgs[i]
    imgDir = imgFolder + "/" + imgFile
    img = Image.open(imgDir)
    #print("IMG OPJECT: " + str(img))
    labelsDir = imgDir.replace("images", "labels").replace('jpg','txt')
    width, height = img.size
    #print("LABEL: " + labelsDir)
    
    negsFromImg = 0

    w = random.randrange(100,230)
    h = random.randrange(100,230)
    centX = random.randrange(w//2, width - w // 2)
    centY = random.randrange(h//2, height - h // 2)

    randCrop = (centX - w // 2, centY - h // 2, centX + w//2, centY + h//2)

    overlap = False
    
    width, height = img.size

    imgID = -1
    for imgObj in jsonData['images']:
        if imgObj['file_name'] == imgFile:
            imgID = imgObj['id']
            break
    dims = None
    for annObj in jsonData['annotations']:
        if annObj['image_id'] == imgID:
            bbox = annObj['bbox']
            dims = (bbox[0],bbox[1],bbox[0]+bbox[1],bbox[1]+bbox[3])
            
            if do_boxes_overlap(dims, randCrop):
                overlap = True
                break
            
    if not overlap:
        croppedImg = img.crop(randCrop)
        croppedImg.save(outDir + "/" + str(negCount) + ".jpg")
        negsFromImg += 1
        negCount += 1

    if negCount >= 3850:
        break

    
