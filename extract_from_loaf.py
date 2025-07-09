from os import listdir
from PIL import Image

dirArr = ['/train','/val','/test']
imgFolder = 'loaf_data/images' + dirArr[0]

loafImgs = listdir(imgFolder)

personCount = 0
outDir = 'loaf_data/extracts/train/person'

for imgFile in loafImgs:
    imgDir = imgFolder + "/" + imgFile
    img = Image.open()
    labelsDir = imgDir.replace("images", "labels").replace('jpg','txt')
    print("LABEL: " + labelsDir)
    with open(labelsDir, "r") as labelFile:
        line = labelFile.readline()
        while line != "":
            annot = line.rstrip("\n").rsplit(" ")
            centX, centY, w, h = annot[1:]

            #scale according to img dimensions
            width, height = img.size
            centX *= width
            centY *= height
            w *= width
            h *= height

            img = img.crop((centX - w / 2, centY - h / 2, centX + w/2, centY + h/2))
            img.save(outDir + "/" + personCount + ".jpg")
            line = labelFile.readline()

        
