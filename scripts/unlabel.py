import sys
from random import randint as r
import glob
import cv2
import numpy as np

USAGE = "python unlabel.py /path/to/training/image/directory/"

assert len(sys.argv) > 1, f'Insufficient arguments: {USAGE}'
args = sys.argv[1:] if '.py' in sys.argv[0] else sys.argv[0:]
assert(len(args) == 1), f'Incorrect usage: {USAGE}'

IMAGE_DIR = args[0]
images = glob.glob(f'{IMAGE_DIR}/*.jpg')
labels = glob.glob(f'{IMAGE_DIR}/*.txt')
labelled = zip(images[:len(labels)], labels[:len(images)])

classes = open(f'{IMAGE_DIR}/custom.names').readlines()
colors = [(r(0,255), r(0,255), r(0,255)) for c in classes]
for i in range(len(classes)):
    print(f'{classes[i][:-1]}: {colors[i]}')

for image, label in labelled:
    print(f'{image} --> {label}')
    with open(label) as f:
        img = cv2.imread(image)
        height, width = img.shape[:2]
        for line in f.readlines():
            a, x, y, w, h = line.split(' ')
            a, x, y, w, h = int(a), float(x), float(y), float(w), float(h)
            x, y, w, h = x * width, y * height, w * width, h * height
            x1 = int(x - w/2)
            x2 = int(x + w/2)
            y1 = int(y - h/2)
            y2 = int(y + h/2)
            print(f'{classes[a][:-1]} : ({x1},{y1}) , ({x2},{y2})')
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colors[a], 50)
    cv2.imwrite(f'''{image.split('.jpg')[0]}_unlabelled.jpg''', img)