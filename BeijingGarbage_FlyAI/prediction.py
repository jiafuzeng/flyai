# -*- coding: utf-8 -*
import os
from path import MODEL_PATH, DATA_PATH
import numpy as np
from flyai.framework import FlyAI
from keras.models import load_model
from keras_efficientnets import custom_objects
from path import DATA_PATH, MODEL_PATH
import cv2
import sys

input_size = 128

class Prediction(FlyAI):
    def __init__(self):
        self.model = None

    def find_new_file(self,dir):
        '''查找目录下最新的文件'''
        file_lists = os.listdir(dir)
        file_lists.sort(key=lambda fn: os.path.getmtime(dir + "/" + fn))
        file = os.path.join(dir, file_lists[-1])
        return file
    
    def load_model(self):
        '''
        模型初始化，必须在构造方法中加载模型
        '''
        model_path = self.find_new_file(MODEL_PATH)
        if self.model is None:
            self.model = load_model(model_path)

    def predict(self, image_path):
        '''
        模型预测返回结果
        :param image_path: 评估传入样例 {"image_path": "./data/input/image/0.png"}
        :return: 模型预测成功中户 {"label": 0}
        '''
        imagePath = os.path.join(sys.path[0] , image_path[2:])

        img = cv2.imread(imagePath)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_size, input_size))
        img = img / 255.0  # 进行归一化
        batch_x = np.reshape(img, (-1, img.shape[0], img.shape[1], img.shape[2]))
        pred = self.model.predict(batch_x)[0]
        pred_label = np.argmax(pred)
        return {"label": pred_label}

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    pre = Prediction()
    pre.load_model()
    print(pre.predict("./data/input/BeijingGarbage/images/3127.jpeg"))
    pass