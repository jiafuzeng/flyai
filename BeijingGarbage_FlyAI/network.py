from keras.layers import Flatten, Dense, AveragePooling2D
from keras.utils import multi_gpu_model
from Groupnormalization import GroupNormalization
from keras_efficientnets import EfficientNetB5,EfficientNetB7
from keras.layers import Dense,Input,Conv2D, Dropout, Activation,GlobalAveragePooling2D,LeakyReLU,BatchNormalization
from keras.models import  Model
from keras.optimizers import RMSprop,Nadam
from flyai.utils import remote_helper
## incpetionV3
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras import backend as K
from keras.initializers import RandomNormal
from keras.applications import DenseNet201
#from focal_loss import focal_loss,multi_category_focal_loss2_fixed
#from focal_loss import SparseCategoricalFocalLoss
from models.model_factory import get_model
#from tensorflow_addons.losses.focal_loss import SigmoidFocalCrossEntropy


def efficientNetB5Model(learning_rate,input_size,classes):
        objective = 'categorical_crossentropy'
        metrics = ['accuracy']

        #optimizer = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        #optimizer = RMSprop(lr=learning_rate, rho=0.9,epsilon=None,decay=0.001)

        model = EfficientNetB5(weights=None,
                               include_top = False,
                               input_shape=(input_size,input_size,3),
                               classes=classes,
                               pooling=max)
        weight_path = remote_helper.get_remote_date('https://www.flyai.com/m/efficientnet-b5_notop.h5')
        
        model.load_weights(weight_path)
        for i , layer in enumerate(model.layers):
            if "batch_normalization" in layer.name:
                model.layers[i] = GroupNormalization(groups=32,axis=-1,epsilon=0.00001)

        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.4)(x)
    
        predictions = Dense(classes,activation="softmax")(x)
        model = Model(input=model.inputs,output=predictions)
        #model = multi_gpu_model(model, 1) # 修改成自身需要的GPU数量，4代表用4个GPU同时加载程序
        #model.compile(loss=objective,optimizer=optimizer,metrics=metrics)
        model.compile(optimizer='sgd', loss=focal_loss,metrics=['accuracy'])

        return model


def efficientNetB7Model(learning_rate,input_size,classes):
        objective = 'categorical_crossentropy'
        metrics = ['accuracy']

        optimizer = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        #optimizer = RMSprop(lr=learning_rate, rho=0.9,epsilon=None,decay=0.001)

        model = EfficientNetB7(weights=None,
                               include_top = False,
                               input_shape=(input_size,input_size,3),
                               classes=classes,
                               pooling=max)
        #weight_path = remote_helper.get_remote_data(
        #        'https://www.flyai.com/m/efficientnet-b5_notop.h5') 
        #model.load_weights(weight_path)
        
        #for i , layer in enumerate(model.layers):
        #    if "batch_normalization" in layer.name:
        #        model.layers[i] = GroupNormalization(groups=32,axis=-1,epsilon=0.00001)

        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.4)(x)

        predictions = Dense(classes,activation="softmax")(x)
        model = Model(input=model.inputs,output=predictions)
        #model = Model(input=model.inputs,output=model.output)
        #model = multi_gpu_model(model, 1) # 修改成自身需要的GPU数量，4代表用4个GPU同时加载程序
        model.compile(loss=objective,optimizer=optimizer,metrics=metrics)

        return model



def incpetionV3Model(learning_rate,input_size,classes):
    # 添加全局平均池化层
    # 构建不带分类器的预训练模型
    base_model = InceptionV3(weights=None,
                             input_shape=(input_size,input_size,3),
                             include_top=False,
                             pooling= max)
    path = remote_helper.get_remote_data(
            'https://www.flyai.com/m/v0.5|inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    base_model.load_weights(path)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # 添加全连接层
    x = Dense(1024, activation='relu')(x)
    x = Dense(512,activation='relu')(x)
    # 添加一个分类器
    predictions = Dense(classes, activation='softmax')(x)
    # 构建我们需要训练的完整模型
    model = Model(inputs=base_model.input, outputs=predictions)
    # 编译模型（一定要在锁层以后操作）
    #optimizer = RMSprop(lr=learning_rate, rho=0.9,epsilon=None,decay=0.01)
    #optimizer = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer='sgd', loss=SigmoidFocalCrossEntropy(),metrics=['accuracy'])
    
    return model

def DenseNet201Model(learning_rate,input_size,classes):
    
    # 必须使用该方法下载模型，然后加载
    path = remote_helper.get_remote_data('https://www.flyai.com/m/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')
    
    base_model = DenseNet201(include_top=False,
                             weights=None,
                             input_shape=(input_size,input_size,3),
                             pooling=max,
                             classes=classes)

    base_model.load_weights(path)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # 添加全连接层
    x = Dense(1024, activation='relu')(x)
    x = Dense(512,activation='relu')(x)
    # 添加一个分类器
    predictions = Dense(classes, activation='softmax')(x)
    # 构建我们需要训练的完整模型
    model = Model(inputs=base_model.input, outputs=predictions)
    # 编译模型（一定要在锁层以后操作）
    #optimizer = RMSprop(lr=learning_rate, rho=0.9,epsilon=None,decay=0.01)
    optimizer = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    #model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
    model.compile(optimizer=optimizer, loss=SigmoidFocalCrossEntropy(),metrics=['accuracy'])
    
    return model

def resNestModel(learning_rate,input_size,classes):
     
    fc_activation='softmax' #softmax sigmoid
    model = get_model(model_name="ResNest200",
                      input_shape=[input_size,input_size,3],
                      n_classes=classes,
                      verbose=True,
                      fc_activation=fc_activation,
                      using_cb=False)
    # optimizer = RMSprop(lr=learning_rate, rho=0.9,epsilon=None,decay=0.01)
    # optimizer = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    #model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])     
    model.compile(optimizer='sgd', loss="categorical_crossentropy",metrics=['accuracy'])     
    return model