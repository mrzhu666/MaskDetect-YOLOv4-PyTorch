import os
import cv2
import torch
import colorsys
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm.std import tqdm
from yolo import YOLO
from nets.yolo4 import YoloBody
import xml.etree.ElementTree as ET
from PIL import Image, ImageFont, ImageDraw
from utils.utils import non_max_suppression, bbox_iou, DecodeBox,letterbox_image,yolo_correct_boxes
import yaml

f = open('config/setting.yaml', encoding='utf-8')
param=yaml.safe_load(f)
val_param=param['val']


# 测试集和训练过程中的测试一致
# 生成文件再mAP/input mAP/output

# ground-truth annotations path
annotation_path = param['val']['annotation_path']
annotations = sorted(os.listdir(annotation_path))
# test images path
image_path = param['val']['image_path']
images = sorted(os.listdir(image_path))

# making ground-truth
print('Start making ground-truth!')
for annotation in annotations:
    tree = ET.parse(os.path.join(annotation_path, annotation))
    objects = tree.findall('object')
    with open(os.path.join('mAP/input/ground-truth', annotation.split('.')[0] + '.txt'), 'w+') as f:
        for obj in objects:
            name = str(obj.find('name').text)
            bndbox = obj.find('bndbox')
            xmin = str(bndbox.find('xmin').text)
            ymin = str(bndbox.find('ymin').text)
            xmax = str(bndbox.find('xmax').text)
            ymax = str(bndbox.find('ymax').text)
            f.write(' '.join((name, xmin, ymin, xmax, ymax)) + '\n')
print('Finish making ground-truth!')

# making detection-results
class mAP_Yolo(YOLO):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def detect_image(self, image_id, image):
        self.confidence = 0.05
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(image, (self.model_image_size[0], self.model_image_size[1])))
        photo = np.array(crop_img, dtype = np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)
        images = np.asarray(images)

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            
        output_list = []
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i]))
        output = torch.cat(output_list, 1)
        batch_detections = non_max_suppression(output, len(self.class_names),
                                                conf_thres=self.confidence,
                                                nms_thres=0.3)

        try:
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return image
            
        top_index = batch_detections[:,4]*batch_detections[:,5] > self.confidence
        top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
        top_label = np.array(batch_detections[top_index,-1],np.int32)
        top_bboxes = np.array(batch_detections[top_index,:4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

        # 去掉灰条
        boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)

        with open(os.path.join('mAP/input/detection-results', image_id + '.txt'), 'w+') as f:
            for i, c in enumerate(top_label):
                predicted_class = self.class_names[c]
                score = str(top_conf[i])

                top, left, bottom, right = boxes[i]
                f.write(' '.join((str(predicted_class), str(score), str(left), str(top), str(right), str(bottom))) + '\n')
    

print('Start making detection results!')
yolo = mAP_Yolo(model_path=val_param["model_path"])
for image in tqdm(images): 
    img = Image.open(os.path.join(image_path, image))
    yolo.detect_image(image.split('.')[0], img)
    # print('[done] ' + image)
print('Finish making detection results!')