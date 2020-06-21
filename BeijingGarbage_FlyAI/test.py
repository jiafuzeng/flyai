import pandas as pd
import os
from path import DATA_PATH
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

NB_CLASSES = 214
DATA_ID = 'BeijingGarbage'

df = pd.read_csv(os.path.join(DATA_PATH, DATA_ID, 'train.csv'))
image_path_list = df['image_path'].values
label_list = df['label'].values

i = 1        
    
print(len(image_path_list))
kf = KFold(n_splits=100,random_state=233,shuffle=True)
for train_indix,val_indix in kf.split(df['image_path'].values):
    print(i)
    train_image_path_list,val_image_path_list = image_path_list[train_indix],image_path_list[val_indix]
    train_label_list,val_label_list = label_list[train_indix],label_list[val_indix]
    print("train: ", len(train_image_path_list))
    print("val: ",len(val_image_path_list))
    i+=1