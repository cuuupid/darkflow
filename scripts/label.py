import sys, os
import json
import glob
from PIL import Image

USAGE = "python label.py /path/to/training/image/directory/ /path/to/labelbox/exported/xy/data.json"

assert len(sys.argv) > 2, f'Insufficient arguments: {USAGE}'
args = sys.argv[1:] if '.py' in sys.argv[0] else sys.argv[0:]
assert(len(args) == 2), f'Incorrect usage: {USAGE}'

IMAGE_DIR = args[0]
JSON_FILE = args[1]

with open(JSON_FILE) as f:
    data = json.load(f)
    print(f'Found {len(data)} datapoints.')
images = glob.glob(f'{IMAGE_DIR}/*.jpg')
print(f'Found {len(images)} images.')

classes = list(set(str(key) for d in data if type(d["Label"]) == dict for key in d["Label"].keys()))
start = 'START' in classes
if start: classes.remove('START')
print(f'Found {len(classes)} classes.')

def convert_bbox(bbox, width, height, offset_x=0, offset_y=0):
    x = min([coord["x"] for coord in bbox])
    y = min([coord["y"] for coord in bbox])
    w = max([coord["x"] for coord in bbox]) - x
    h = max([coord["y"] for coord in bbox]) - y
    x += w/2
    y += h/2
    x -= max(offset_x, 0)
    y = max(offset_y - y, 0)
    return x/width, y/height, w/width, h/height

for image in images:
    # Trim the extension for our use case
    image = image.split('/')[-1]
    # Get the width and height of the original image
    img = Image.open(f'{IMAGE_DIR}/{image}')
    width, height = img.size[:2]
    width, height = int(width), int(height)
    # Next we look up the image in the JSON data
    annotation = [d for d in data if image.split(' ')[0] in d["Labeled Data"]][0]
    with open(f'{IMAGE_DIR}/{image[:image.rfind(".")]}.txt', 'w') as label:
        if start:
            startbox = annotation["Label"]["START"][0]
            offset_x, offset_y = startbox["x"], startbox["y"]
        else: offset_x, offset_y = 0, height
        if type(annotation["Label"]) != dict:
            continue
        for bboxset in annotation["Label"].keys():
            if bboxset == "START": continue
            for bbox in annotation["Label"][bboxset]:
                x, y, w, h = convert_bbox(bbox, width, height, offset_x, offset_y)
                label.write(f'{classes.index(bboxset)} {x} {y} {w} {h}\n')
    print(f'{image}: {sum([len(annotation["Label"][k]) for k in annotation["Label"].keys()])} annotations')

with open(f'{IMAGE_DIR}/train.txt', 'w') as image_paths:
    for image in images:
        image_paths.write(f'{image}\n')

with open(f'{IMAGE_DIR}/custom.names', 'w') as names:
    for classname in classes:
        names.write(f'{classname}\n')

with open(f'{IMAGE_DIR}/custom.cfg', 'w') as config:
    config.write(f'classes={len(classes)}\n')
    config.write(f'train={IMAGE_DIR}/train.txt\n')
    config.write(f'valid={IMAGE_DIR}/train.txt\n') # TODO: add support for validation set
    config.write(f'names={IMAGE_DIR}/custom.names\n')
    config.write(f'backup=backup') # TODO: add custom checkpointing directory

print("Change the classes at the bottom of and in the middle of the architecture configuration you're using to match the number of classes you have.")
print("In the same file also change filters of the first conv. layer, the value should be num/3*(classes+5)")