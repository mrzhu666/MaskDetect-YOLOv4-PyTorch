import os
import xml.etree.ElementTree as ET



# 读取xml的标注数据，存储到model_data中

# common parameters
class_index = ('mask', 'nomask')


# train dataset parameters
images_path = '../../data/MaskDatasets/train/JPEGImages'
annotations_path = '../../data/MaskDatasets/train/Annotations'
# get train dataset list
images = sorted(os.listdir(images_path))
annotations = sorted(os.listdir(annotations_path))


# make train annotations
with open('model_data/mask_train.txt', 'w+') as f:
    for i in range(len(annotations)):
        annotation_path = annotations[i]
        image_path = images[i]
        f.write(images_path + '/' + image_path + ' ')
        
        tree = ET.parse(annotations_path + '/' + annotation_path)
        objects = tree.findall('object')
        for obj in objects:
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            f.write(str(bndbox.find('xmin').text) + ',' +
                    str(bndbox.find('ymin').text) + ',' +
                    str(bndbox.find('xmax').text) + ',' +
                    str(bndbox.find('ymax').text) + ',' +
                    str(class_index.index(name)) + ' ')
        f.write('\n')


# train dataset parameters
images_path = '../../data/MaskDatasets/val/JPEGImages'
annotations_path = '../../data/MaskDatasets/val/Annotations'
# get train dataset list
images = sorted(os.listdir(images_path))
annotations = sorted(os.listdir(annotations_path))


# make train annotations
with open('model_data/mask_val.txt', 'w+') as f:
    for i in range(len(annotations)):
        annotation_path = annotations[i]
        image_path = images[i]
        f.write(images_path + '/' + image_path + ' ')
        
        tree = ET.parse(annotations_path + '/' + annotation_path)
        objects = tree.findall('object')
        for obj in objects:
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            f.write(str(bndbox.find('xmin').text) + ',' +
                    str(bndbox.find('ymin').text) + ',' +
                    str(bndbox.find('xmax').text) + ',' +
                    str(bndbox.find('ymax').text) + ',' +
                    str(class_index.index(name)) + ' ')
        f.write('\n')