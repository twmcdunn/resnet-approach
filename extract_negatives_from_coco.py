import os
from torchvision.datasets import CocoDetection
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

# Paths
coco_root = './coco'
ann_file = os.path.join(coco_root, 'annotations/instances_train2017.json')
img_dir = os.path.join(coco_root, 'train2017')
output_dir = './coco_personal_belongings'

assert os.path.exists(ann_file), "CAN't find annotation file"
assert os.path.exists(img_dir), "cant find image directory"
print("Directories and files found.")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load COCO annotations
coco = COCO(ann_file)

# Define personal belongings categories
target_categories = ['backpack', 'handbag', 'suitcase', 'chair', 'umbrella', 'laptop', 'cell phone', 'book', 'bottle']
category_ids = coco.getCatIds(catNms=target_categories)
print("Category IDs:" + str(category_ids))
assert len(category_ids) > 0, "Can't find categories!"


categories = coco.loadCats(coco.getCatIds())
print([cat['name'] for cat in categories])


# Get image IDs that contain at least one of the target categories
image_ids_with_targets = set()
for cat_id in category_ids:
    image_ids = coco.getImgIds(catIds=[cat_id])
    image_ids_with_targets.update(image_ids)

image_ids_with_person = set(coco.getImgIds(coco.getCatIds(catNms=['person'])))

image_ids = image_ids_with_targets - image_ids_with_person

print(f"Found {len(image_ids)} images with target categories.")
assert len(image_ids) > 0, "Can't find images with target categories!"

#assert 0 == 1, "EXITING SO YOU CAN LOOK FOR PEOPLE IN CATEGORIES"

# Define transform: resize with preserved aspect ratio, then random crop
transform = transforms.Compose([
    transforms.Resize(250),  # Resize the vertical to 250, preserving aspect ratio
    transforms.RandomCrop((155, 204)),  # Final target size
])

# Process and save images
for img_id in tqdm(image_ids, desc="Extracting and transforming images"):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, img_info['file_name'])
    try:
        img = Image.open(img_path).convert('RGB')
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if ann['category_id'] in category_ids:
                x,y,w,h = ann['bbox']
                cropped = img.crop((x,y,x+w,y+h))
                cropped = transform(cropped)
                cropped.save(os.path.join(output_dir, str(ann['category_id']) + "_" + img_info['file_name']))
    except Exception as e:
        print(f"Failed to process {img_path}: {e}")
