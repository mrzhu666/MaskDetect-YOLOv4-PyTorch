import cv2
import time
import numpy as np
from yolo import YOLO
from PIL import Image


def main():
    image_path='./data/val_271.jpg'

    print('Start detect!')
    yolo = YOLO()
    try:
        image = Image.open(image_path)
    except:
        print('Open Error! Try again!')
        pass
    else:
        r_image = yolo.detect_image(image)
        r_image.save(image_path.split('.')[0] + '_result.png')
    print('Finish detect!')


if __name__=='__main__':
    main()

