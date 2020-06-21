# -*- coding: utf-8 -*-
from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
import pandas as pd
from path import DATA_PATH, MODEL_PATH
import os
import argparse
import cv2
import numpy as np
import math
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from flyai.utils import remote_helper
from flyai.utils.log_helper import train_log
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from network import efficientNetB5Model,incpetionV3Model,efficientNetB7Model,resNestModel,DenseNet201Model
from warmup_cosine_decay_scheduler import WarmUpCosineDecayScheduler
from keras.callbacks import TensorBoard, Callback,ModelCheckpoint
#from tensorflow.keras.callbacks import ModelCheckpoint,Callback,TensorBoard
import tensorflow as tf


'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''

NB_CLASSES = 214
DATA_ID = 'BeijingGarbage'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
pathabs = os.path.abspath(__file__)
train_path = "{}/train_path".format(os.path.dirname(pathabs))
if not os.path.exists(train_path):
    os.makedirs(train_path)

log_path = "{}/logs".format(os.path.dirname(pathabs))
if not os.path.exists(log_path):
    os.makedirs(log_path)

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=30, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=64, type=int, help="batch size")
parser.add_argument("--train_path", default= train_path, type=str, help="train path")
parser.add_argument("--learning_rate", default=0.5, type=float, help="learning rate")
parser.add_argument("--input_size", default=128, type=int, help="image input size")
parser.add_argument("--log_path", default=log_path, type=str, help="log path")
parser.add_argument("--num_save", default=10, type=int, help="Epoch saves the model every few times")
args = parser.parse_args()


def get_batch_data(image_path_list, label_list, batch_size,input_size):
    batch_x = []
    batch_y = []
    while len(batch_x) < batch_size:
        index = np.random.randint(len(image_path_list))
        img = cv2.imread(os.path.join(DATA_PATH, DATA_ID, image_path_list[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = np.zeros(NB_CLASSES)
        label[label_list[index]] = 1
        img = cv2.resize(img, (input_size, input_size))
        batch_x.append(img)
        batch_y.append(label)
    return np.array(batch_x), np.array(batch_y)

def train_batch_data(batch_step,image_path_list, label_list, batch_size,input_size):
    train_start = batch_step * batch_size
    train_end = (batch_step + 1) * batch_size
    dataSize = len(image_path_list)
    if  train_end <= dataSize:
        train_data = image_path_list[train_start:train_end]
        train_label = label_list[train_start:train_end]
    else :
        train_data = image_path_list[train_start:]
        train_label = label_list[train_start:]
    
    batch_x = []
    for path in train_data:
        img = cv2.imread(os.path.join(DATA_PATH, DATA_ID, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_size, input_size))
        batch_x.append(img)
    train_label = np_utils.to_categorical(train_label,NB_CLASSES)
    return np.array(batch_x), np.array(train_label)
                

class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''

    def download_data(self):
        # 下载数据
        data_helper = DataHelper()
        # ./data/input/xxxx/
        data_helper.download_from_ids(DATA_ID)

    def deal_with_data(self):
        '''
        处理数据，没有可不写。
        :return:
        '''
        # 数据增强
        train_datagen = ImageDataGenerator(rescale=1. / 255,  # 归一化
                                           zca_whitening=False, #应用zca白话
                                           rotation_range=10,  # 旋转角度
                                           width_shift_range=0.1,  # 水平偏移
                                           height_shift_range=0.1,  # 垂直偏移
                                           shear_range=0.1,  # 随机错切变换的角度
                                           zoom_range=0.1,  # 随机缩放的范围
                                           horizontal_flip=False,  # 随机将一半图像水平翻转
                                           fill_mode='nearest')  # 填充像素的方法

        val_datagen = ImageDataGenerator(rescale=1. / 255)
        return train_datagen, val_datagen

    def smooth_labels(self,labels, factor=0.1):
        # smooth the labels
        labels *= (1 - factor)
        labels += (factor / labels.shape[1])

        # returned the smoothed labels
        return labels
    
    def train(self):
        # 读取数据
        df = pd.read_csv(os.path.join(DATA_PATH, DATA_ID, 'train.csv'))
        image_path_list = df['image_path'].values
        label_list = df['label'].values

        # 划分训练集和校验集
        all_size = len(image_path_list)
        train_size = int(all_size * 0.9)
        #train_image_path_list = image_path_list[:train_size]
        #train_label_list = label_list[:train_size]
        train_image_path_list = image_path_list
        train_label_list = label_list
        val_image_path_list = image_path_list[train_size:]
        val_label_list = label_list[train_size:]
        valdataSize = len(val_image_path_list)
        trainDataSize = len(train_image_path_list)
        print('train_size: %d, val_size: %d' % (trainDataSize, valdataSize))
        # 构建模型
        batch_steps = math.ceil(trainDataSize/ args.BATCH)
        # 训练模型
        model = incpetionV3Model(learning_rate=args.learning_rate,input_size=args.input_size,classes=NB_CLASSES) 
        #model = efficientNetB5Model(learning_rate=args.learning_rate,input_size=args.input_size,classes=NB_CLASSES) 
        #model = DenseNet201Model(learning_rate=args.learning_rate,input_size=args.input_size,classes=NB_CLASSES) 
        #model = resNestModel(learning_rate=args.learning_rate,input_size=args.input_size,classes=NB_CLASSES) 

        warmup_epoch = 5
        total_steps = int(args.EPOCHS * trainDataSize / args.BATCH)
        warmup_steps = int(warmup_epoch * trainDataSize / args.BATCH)
        
        warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=args.learning_rate,
                                            total_steps=total_steps,
                                            warmup_learning_rate=0.1,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=5,
                                            verbose=1
                                            )

        #tensorBoard = TensorBoard(log_dir=args.log_path) 
        
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)

        save_path = os.path.join(MODEL_PATH, 'model_{epoch:03d}-{val_accuracy:.4f}.h5')
        
        checkout = ModelCheckpoint( save_path,
                                    monitor='val_acc', 
                                    verbose=0,
                                    save_best_only=True,
                                    save_weights_only=False, 
                                    mode='auto', 
                                    period=int(args.num_save))

        for epoch in range(args.EPOCHS):
            val_acc_list = []
            for i in range(batch_steps):
                now_step = epoch * batch_steps + i
                
                temp_x, temp_y = train_batch_data(i,train_image_path_list,train_label_list,args.BATCH,args.input_size)

                temp_val_x, temp_val_y = get_batch_data(val_image_path_list, val_label_list,args.BATCH,args.input_size)

                # 数据增强
                train_datagen, val_datagen = self.deal_with_data()
                batch_gen = train_datagen.flow(temp_x, y=temp_y, batch_size=args.BATCH)
                batch_x, batch_y = next(batch_gen)
                batch_val_gen = val_datagen.flow(temp_val_x, y=temp_val_y, batch_size=args.BATCH)
                batch_val_x, batch_val_y = next(batch_val_gen)
                batch_y = self.smooth_labels(batch_y)
                batch_val_y = self.smooth_labels(batch_val_y)

                #history = model.fit(batch_x, batch_y, batch_size=args.BATCH, verbose=0,
                #                    validation_data=(batch_val_x, batch_val_y),callbacks=[warm_up_lr,tensorBoard,checkout])

                history = model.fit(batch_x, batch_y, batch_size=args.BATCH, verbose=0,
                                    validation_data=(batch_val_x, batch_val_y),callbacks=[warm_up_lr,checkout])
                
                train_loss = history.history['loss'][0]
                train_acc = history.history['accuracy'][0]
                val_loss = history.history['val_loss'][0]
                val_acc = history.history['val_accuracy'][0]

                train_log(train_loss=train_loss, train_acc=train_acc, val_loss=val_loss,val_acc=val_acc)
                print(
                    'epoch: %d/%d, batch: %d/%d,now_step: %d ,train_loss: %f, val_loss: %f, train_acc: %f, val_acc: %f' %
                    (epoch, args.EPOCHS, i, batch_steps,now_step ,train_loss, val_loss, train_acc, val_acc))

                val_acc_list.append(val_acc)
            
            max_acc = np.mean(val_acc_list)
            #path = MODEL_PATH + "/model.h5"
            #model.save(path)
            print("******************** epoll %d, accuracy %g" % (epoch,max_acc))


if __name__ == '__main__':
    print("*" *100, tf.__version__)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    main = Main()
    main.download_data()
    main.train()