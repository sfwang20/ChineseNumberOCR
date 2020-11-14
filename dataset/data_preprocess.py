import numpy as np
import os
from cv2 import cv2 
import random

annotations_path = "./annotations/" 
data_path = "./images/"

file_list = os.listdir(annotations_path)
#data_file_list = os.listdir(data_path)

for file_name_annot in file_list:
    file_name = file_name_annot[:-4]
    file_name_img = data_path + file_name + '.jpg'
    img = cv2.imread(file_name_img, cv2.IMREAD_GRAYSCALE)

    file_path = "./annotations/" + file_name_annot
    with open(file_path) as f:           # 讀annot .txt每一行
        annots = f.readlines()
    # 小心如果在win10會有"\n" 需去掉"\n" (最後一行沒有)
    annots[-1] = annots[-1] + "\n"
    annots = [annot[:-1] for annot in annots]

    label_cnt = np.zeros(11, dtype = int)
    if (random.randint(1,10) > 8):
        data_set = 'test/'
    elif (random.randint(1,10) > 7):
        data_set = 'val/'
    else:
        data_set = 'train/'
    for annot in annots:
        data = annot.split(" ") 
        label = str(int(data[0]) + 1)
        w = int(float(data[3]) * img.shape[1])
        h = int(float(data[4]) * img.shape[0])
        x = int(float(data[1]) * img.shape[1] - w/2)
        y = int(float(data[2]) * img.shape[0] - h/2)
        # Crop image
        img_crop = img[y:y+h, x:x+w]
        img_crop = cv2.resize(img_crop, (128, 128))
        out_file_path =  data_set + label + '/' 
        
        if not os.path.exists(out_file_path):
            os.mkdir(out_file_path)
        
        cv2.imwrite(out_file_path + file_name + '_' + str(label_cnt[int(label)]) + '.jpg', img_crop)
        label_cnt[int(label)] += 1

        print(file_name_img + "...Done!")