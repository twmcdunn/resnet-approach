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

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load COCO annotations
coco = COCO(ann_file)

# Define personal belongings categories
target_categories = ['backpack', 'handbag', 'suitcase', 'umbrella', 'laptop', 'cell phone', 'book', 'bottle']
category_ids = coco.getCatIds(catNms=target_categories)

# Get image IDs that contain at least one of the target categories
image_ids = coco.getImgIds(catIds=category_ids)

# Define transform: resize with preserved aspect ratio, then random crop
transform = transforms.Compose([
    transforms.Resize(180),  # Resize shorter side to 180, preserving aspect ratio
    transforms.RandomCrop((155, 204)),  # Final target size
])

# Process and save images
for img_id in tqdm(image_ids, desc="Extracting and transforming images"):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, img_info['file_name'])
    try:
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        img.save(os.path.join(output_dir, img_info['file_name']))
    except Exception as e:
        print(f"Failed to process {img_path}: {e}")
