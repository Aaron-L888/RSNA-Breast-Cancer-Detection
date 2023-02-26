#!pip install -qU python-gdcm pydicom pylibjpeg

# Data manipulation and visualization libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import cv2
import tensorflow as tf
# File and directory handling libraries
import pydicom
from os import listdir
import dicomsdl
import math

# Statistical and data processing libraries
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight
from scipy.stats import mode, skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from fancyimpute import KNN
# Progress tracking and warning suppression libraries
from tqdm.notebook import trange
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set seaborn style
sns.set_style('darkgrid')

###RSNA Breast Cancer Detection Model - Kaggle
#Solution Format: 
#  prediction_id,cancer
#  10008-L,0
#  10008-R,0.5
#  10009-L,1
#  ...

##Bring in Patient Metadata and see format:
filepath = 'datasets/rsna-bcd/train.csv'
data = pd.read_csv(filepath)
print(data.head())
print(data.info()) #three features appear to have missing data

#Add image path to patient metadata df
data["image_path"] = 'datasets/rsna-bcd/train_images/'+data["patient_id"].astype(str) +"/"+ data["image_id"].astype(str)+".dcm"

#image location format
images_dir = 'datasets/rsna-bcd/{}_images/{}/{}.dcm' 

#Verify there are missing values:
print(data.isna().sum()) #missing values only for age, BIRADS and density

#Cancer and Biopsy Rates
biopsy_counts = data.groupby('cancer')['biopsy'].value_counts().unstack().fillna(0)
biopsy_perc = biopsy_counts.transpose() / biopsy_counts.sum(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
data['cancer'].value_counts().plot.bar(ax=axes[0])
sns.heatmap(biopsy_perc, square=True, annot=True, fmt='.1%', cmap='Blues', ax=axes[1])
axes[0].set_title("Number of images showing cancer")
axes[1].set_title("Percentage of images\nresulting in a biopsy")
plt.show()

#Examine Age Distribution
all_ages = data.groupby('patient_id')['age'].apply(lambda x: x.unique()[0])
cancer_ages = data[data['cancer'] == 1].groupby('patient_id')['age'].apply(lambda x: x.unique()[0])

plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
sns.histplot(all_ages, bins=64, color='blue', kde=True)
plt.title("All Patients")
plt.xlim(33, 89)
plt.subplot(1, 2, 2)
sns.histplot(cancer_ages, bins=50, color='red', kde=True)
plt.title("Patients with Cancer")
plt.xlim(33, 89)
plt.suptitle("Age Distribution of Patients")
plt.show()

# Age Statistics
print("Mean Age:", all_ages.mean().round(1))
print("Age Std:", all_ages.std().round(1))
print("Age Q1:", all_ages.quantile(0.25))
print("Median Age:", all_ages.median())
print("Age Q3:", all_ages.quantile(0.75))

#Examine Density and Difficult Cases:
fig, axes = plt.subplots(ncols=2, figsize=(12, 7))
data['difficult_negative_case'].value_counts().plot.bar(ax=axes[0])
data['density'].value_counts().plot.bar(ax=axes[1])
axes[0].set_title("Difficult Cases to Diagnose")
axes[1].set_title("Density Categories")
plt.suptitle("Summay Data")
plt.show()

#Overall Data Correlation
corr_df = data.drop(columns=['patient_id', 'image_id', 'site_id', 'machine_id'])
fig, ax = plt.subplots(figsize=(16, 9))
dataplot = sns.heatmap(corr_df.corr(method='spearman'), cmap="YlGnBu", annot=True)
plt.suptitle("Overall Correlation")
plt.show()
#Biopsy strongly correlated with cancer, Age weakly correlated


##Data Preprocessing
#image location format
images_dir = 'datasets/rsna-bcd/{}_images/{}/{}.dcm' 

#Filling in missing values
mean_imputer = SimpleImputer(strategy='mean')
data['age'] = mean_imputer.fit_transform(data[['age']]) #missing age values filled in with mean

#Drop BIRADS and density...large number of missing data, not available in test csv.
#large number of missing BIRADS and density values, use fancyimputer module to fill:
#creating maps of integers for KNN to use
#data['BIRADS'] = data['BIRADS'].map({'0': 0, '1': 1, '2': 2})
#data['density'] = data['density'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})

# Impute missing values for BIRADS and density using KNN imputation
#imputed_data = KNN(k=5).fit_transform(data[['BIRADS', 'density']]) #filling with values from 5 nearest neighbors, changes fit_transform to complete()
#data[['BIRADS', 'density']] = imputed_data #code wasn't reachable with fit_transform

# Convert imputed numerical columns back to categorical representation
#data['BIRADS'] = data['BIRADS'].map({0: '0', 1: '1', 2: '2'})
#data['density'] = data['density'].map({0: 'A', 1: 'B', 2: 'C', 3: 'D'})
#slow impute calculation

##split data
data_downsample = data[data['cancer'] == 1] 
data_null = data[data['cancer'] == 0].sample(1242)
data_ds_combined = pd.concat([data_downsample, data_null], axis=0).sample(frac=1, random_state=42)
data_train = data_ds_combined.sample(1400)
print(data_train['cancer'].head(5))
#print(data_downsample.info())
#print(data_ds_combined.head(10))

X_train, X_val = train_test_split(data_train, test_size = 0.2, random_state = 42)
print(len(X_train), len(X_val)) #(38294 16412)

#Image params
img_height = 512
img_width = 256
img_shape = (img_height, img_width, 3)

#Image Data Class
class ImageData(tf.keras.utils.Sequence): #subclassed from Sequence: Base object for fitting to a sequence of data, such as a dataset
        def __init__(self, df, batch_size, mode_str): 
            self.df = df
            self.batch_size = batch_size
            self.mode_str = mode_str
            self.len = len(df)
        
        def __getitem__(self, index):
            start, end = index * self.batch_size, (index + 1) * self.batch_size #for example 32:64
            X = np.zeros((self.batch_size, ) + img_shape) #N batches of arrays, each 300x250x1  so each pixel is an inner value
            y = np.zeros((self.batch_size, 1)) #length of batch_size, 2D so each value is inner array
            
            for i , pos in enumerate(range(start, end)):
                if pos >= self.len: break 
                        
                row = self.df.iloc[pos]
                patient_id = row.patient_id
                img_id = row.image_id
                
                if self.mode_str == "train" or "valid":
                    file_name = images_dir.format("train", patient_id, img_id)
                else:
                    file_name = images_dir.format("test", patient_id, img_id)

                img = dicomsdl.open(file_name)
                img_arr = img.pixelData()
                
                # Standardize each scan
                img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
                img_arr *= 255

                if img.PhotometricInterpretation == "MONOCHROME1":
                    img_arr = img_arr.max() - img_arr
                if img.PhotometricInterpretation == "MONOCHROME2":
                    img_arr = img_arr - img_arr.min()

                img_arr = np.expand_dims(img_arr, axis = -1) #axis=-1 just adds new column at the end
                img_arr = tf.image.resize_with_pad(img_arr, img_height, img_width, method = 'nearest').numpy() #img_shape[:-1] for resize

                if self.mode_str == "train":
                    seedi = (pos, 0)
                    img_arr = tf.image.stateless_random_contrast(img_arr, 0.8, 1.2, seed=seedi)
                    img_arr = tf.image.stateless_random_flip_left_right(img_arr, seed=seedi)
                    #img_arr = tf.image.stateless_random_saturation(img_arr, 0.5, 1.0, seed=seedi)
                    img_arr = tf.image.stateless_random_brightness(img_arr, 0.2, seed=seedi)

                X[i,...] = img_arr 
                if self.mode_str == "train" or "valid":
                    y[i] = row.cancer
                    
            return (X, y) if self.mode_str == "train" or "valid" else X
       
        def __len__(self):
            return math.ceil(self.len / self.batch_size)

#Image datasets with target added
train_gen = ImageData(X_train, 32, "train")

val_gen = ImageData(X_val, 32, "valid")

#Preparing metadata
X_train_num = X_train.drop(columns=['patient_id', 'image_id', 'site_id', 'machine_id', 'BIRADS', 'density', 'difficult_negative_case'])
X_val_num = X_val.drop(columns=['patient_id', 'image_id', 'site_id', 'machine_id', 'BIRADS', 'density', 'difficult_negative_case'])

#X_train_num['BIRADS'] = X_train_num['BIRADS'].map({'0': 0, '1': 1, '2': 2})
#X_train_num['density'] = X_train_num['density'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})
#X_train_num['difficult_negative_case'] = X_train_num['difficult_negative_case'].map({"False": 0, "True": 1})
#X_val_num['BIRADS'] = X_val_num['BIRADS'].map({'0': 0, '1': 1, '2': 2})
#X_val_num['density'] = X_val_num['density'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})
#X_val_num['difficult_negative_case'] = X_val_num['difficult_negative_case'].map({"False": 0, "True": 1})

X_train_num_target = X_train_num[['cancer']]
X_train_num = X_train_num.drop(columns=['cancer'])
X_val_num_target = X_val_num[['cancer']]
X_val_num = X_val_num.drop(columns=['cancer'])


#Building model
tf.random.set_seed(42)

base_resnet = tf.keras.applications.resnet.ResNet50(weights='imagenet', include_top=False,
                                                 input_shape=img_shape, pooling = 'avg')
flatten = tf.keras.layers.Flatten()(base_resnet.output)
fc1 = tf.keras.layers.Dense(128, kernel_initializer="he_normal", use_bias=False)(flatten)
bn1 = tf.keras.layers.BatchNormalization()(fc1) 
act1 = tf.keras.layers.Activation("relu")(bn1) 
do1 = tf.keras.layers.Dropout(rate=0.5)(act1)
fc2 = tf.keras.layers.Dense(1, activation ="sigmoid", dtype='float32')(do1)
model_resnet = tf.keras.Model(inputs=base_resnet.input, outputs=fc2)

#Initially freeze resnet layers to train dense layers
for layer in base_resnet.layers:
    layer.trainable = False

optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.01)  
model_resnet.compile(optimizer=optimizer1, loss=tf.keras.losses.BinaryCrossentropy(), 
                  metrics = [tf.keras.metrics.BinaryAccuracy(name='BinAcc')])

#class_weights = dict(zip(np.unique(X_train_num_target), class_weight.compute_class_weight('balanced', classes=np.unique(X_train_num_target), 
#                                                                                            y=X_train_num_target.values.reshape(-1)))) 

history1 =  model_resnet.fit(train_gen, validation_data = val_gen, epochs = 15)

#unfreeze top portion of resnet layers:
for layer in base_resnet.layers[46:]:
    layer.trainable = True

#complete cnn base training
optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.005)
model_resnet.compile(optimizer=optimizer2, loss= tf.keras.losses.BinaryCrossentropy(), 
                metrics = [tf.keras.metrics.AUC(name='AUC'), tf.keras.metrics.BinaryAccuracy(name='BinAcc')])

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("rsna_checkpoints", save_weights_only=True)

history1 =  model_resnet.fit(train_gen, validation_data = val_gen,
                    epochs = 50, callbacks = [lr_scheduler, early_stopping_cb, checkpoint_cb])

#plot results
history_df = pd.DataFrame(history1.history)
history_df[['loss', 'val_loss']].plot()
history_df = pd.DataFrame(history1.history)
history_df[['BinAcc', 'val_BinAcc']].plot()
plt.show()

##Bring in Test Metadata:
filepath2 = 'datasets/rsna-bcd/test.csv'
test_data = pd.read_csv(filepath2)

#Add test image path to patient metadata df
test_data["image_path"] = 'datasets/rsna-bcd/test_images/'+data["patient_id"].astype(str) +"/"+ data["image_id"].astype(str)+".dcm"

test_gen = ImageData(test_data, 32, "test")

pred = model_resnet.predict(test_gen)

test_data['cancer'] = pred[:len(test_data)]
print(test_data.head())

submit = test_data.groupby('prediction_id')['cancer'].max().to_frame().reset_index()
print(submit.head())

submit.to_csv("submission.csv", index = False)


#save model:
model_resnet.save("resnet_rsna_model", save_format="tf")

#load model:
model_resnet = tf.keras.models.load_model("resnet_rsna_model")


# load pre-trained ensemble members
#n_members = 4
#models = list()
#for i in range(n_members):
 # load model
 #filename = 'model_' + str(i + 1) + '.h5'
 #model = load_model(filename)
 # store in memory
 #models.append(model)

# make predictions
#y_hats = [model.predict(testX) for model in models]
#y_hats = np.array(y_hats)
# calculate average
#outcomes = mean(yhats)




