from os import listdir
from PIL import Image
import random

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
for n in range(3):
    imgFolder = 'loaf_data/images' + dirArr[n]

    loafImgs = listdir(imgFolder)

    personCount = 0
    outDir = 'loaf_data/extracts/'+dirArr[n]+'/not_person'

    for i in range(len(loafImgs)):
        if i % 10000 == 0:
            print(str(i/len(loafImgs)) + ' imgs in ' + dirArr[n] + ' negatives')
        imgFile = loafImgs[i]
        imgDir = imgFolder + "/" + imgFile
        img = Image.open(imgDir)
        #print("IMG OPJECT: " + str(img))
        labelsDir = imgDir.replace("images", "labels").replace('jpg','txt')
        #print("LABEL: " + labelsDir)
        
        negsFromImg = 0

        while negsFromImg < 0.01: 
            centX = random.random()
            centY = random.random()
            w = 0.5#random.random()
            h = 0.5#random.random()

            centX *= width
            centY *= height
            w *= width
            h *= height
            w = max(w,230)
            h = max(h, 230)
            randCrop = (centX - w / 2, centY - h / 2, centX + w/2, centY + h/2)

            overlap = False
            
            width, height = img.size
            with open(labelsDir, "r") as labelFile:
                line = labelFile.readline()
                while line != "":
                    annot = line.rstrip("\n").rsplit(" ")
                    centX, centY, w, h = map(float,annot[1:])

                    #scale according to img dimensions
                    centX *= width
                    centY *= height
                    w *= width
                    h *= height

                    dims = (centX - w / 2, centY - h / 2, centX + w/2, centY + h/2)
                    if do_boxes_overlap(dims, randCrop):
                        overlap = True
                        break
                    
                    line = labelFile.readline()
            if not overlap:
                croppedImg = img.crop(randCrop)
                croppedImg.save(outDir + "/" + str(negCount) + ".jpg")
                negsFromImg += 1
                negCount += 1

        if negCount >= 1500:
            break

        
