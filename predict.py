import cv2
import time
import numpy as np
from yolo import YOLO
from PIL import Image
from setting import config

def main():
    image_path='./data/'
    file='val_359.jpg'
    print('Start detect!')
    yolo = YOLO()
    try:
        image = Image.open(image_path+file)
    except:
        print('Open Error! Try again!')
        pass
    else:
        r_image = yolo.detect_image(image)
        file_result=file.split('.')
        file_result[0]+='_result'
        
        r_image.save(image_path+'.'.join(file_result))
    print('Finish detect!')


if __name__=='__main__':
    main()

