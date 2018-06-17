import sys, os
import json
import glob
from PIL import Image

USAGE = "python df_label.py /path/to/training/image/directory/ /path/to/labelbox/exported/xy/data.json"

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

classes = list(set(str(key) for d in data for key in d["Label"].keys()))
classes.remove('START')
print(f'Found {len(classes)} classes.')

#         filename, width, height, x1 x2 y1 y2, class

def convert_bbox(bbox, width, height, offset_x=0, offset_y=0):
    xmin = min([coord["x"] for coord in bbox])
    ymin = min([coord["y"] for coord in bbox])
    xmax = max([coord["x"] for coord in bbox])
    ymax = max([coord["y"] for coord in bbox])
    xmin -= offset_x
    xmax -= offset_x
    ymin = max(offset_y - ymin, 0)
    ymax = max(offset_y - ymax, 0)
    new_width, new_height = 512, 1024
    if height < width: new_width, new_height = new_height, new_width
    xmin = xmin/width*new_width
    xmax = xmax/width*new_width
    ymin = ymin/height*new_height
    ymax = ymax/height*new_height
    return int(xmin), int(xmax), int(ymin), int(ymax), new_width, new_height

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
        startbox = annotation["Label"]["START"][0]
        offset_x, offset_y = startbox["x"], startbox["y"]
        for bbox in annotation["Label"].keys():
            if bbox == "START": continue
            x1, x2, y1, y2, nw, nh = convert_bbox(annotation["Label"][bbox][0], width, height, offset_x, offset_y)
            label.write(f'{image} {nw} {nh} {bbox} {x1} {y1} {x2} {y2}\n')
    print(f'{image}: {len(annotation["Label"].keys())} annotations')

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