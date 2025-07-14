import json

jsonData = None

with open('loaf_data/annotations/annotations/resolution_2k/instances_train.json','r') as f:
    jsonData = json.loads(f.read())

print(str(jsonData))