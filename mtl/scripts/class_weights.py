import os
import numpy as np
from os import listdir
from matplotlib.image import imread
import matplotlib.pyplot as plt

class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
               'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
class_colors = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
                (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]

encoded_colors = []
for tuple in class_colors:
    encoded_colors.append(sum(list(tuple)))
print('encoded colors: ', encoded_colors)

directory = '../../miniscapes/train/semseg'

filenames = [f for f in listdir(directory)]

print(len(filenames))

image_counter = {}
pixel_counter = {}
for name in class_names:
    image_counter[name] = 0
    pixel_counter[name] = 0

for i, name in enumerate(filenames):
    if i % 1000 == 0:
        print('at index ', i)
    path = directory + '/' + name
    image = imread(path)
    image = 255 * image
    image = image.astype(np.uint8)
    #plt.imshow(image)
    #plt.show()
    image = np.sum(image, axis=2)
    unique, counts = np.unique(image, return_counts=True)
    pixel_counts = dict(zip(unique, counts))
    # print(pixel_counts)
    for value in pixel_counts.keys():
        if value == 0:
            #print('number of pixels at with 0 value is ', pixel_counts[value])
            continue
        idx = encoded_colors.index(value)
        object_type = class_names[idx]
        image_counter[object_type] += 1
        pixel_counter[object_type] += pixel_counts[value]

print(pixel_counter)
print(image_counter)

class_freqs = {}
for name in class_names:
    class_freqs[name] = pixel_counter[name] / (360*720*image_counter[name])

print(class_freqs.values())
median_freq = np.median(list(class_freqs.values()))

class_weights = {}
for name in class_names:
    class_weights[name] = median_freq / class_freqs[name]

print(class_weights)