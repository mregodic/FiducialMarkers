import os
import numpy as np # linear algebra
import h5py

import matplotlib
from matplotlib import pyplot as plt

from keras.utils.np_utils import to_categorical

# transform data to a matrix
def data_transform(data):
    data_t = []
    for i in  range(data.shape[0]):
        vector = data[i]
        data_t.append(vector.reshape(38, 38, 38, 1))            
    return np.asarray(data_t, dtype=np.float32)

def histogram_intensity(item, title=''):
        f = []
    
        for i in item:
            b = np.sum(i)
            f.append(b)
     
        n, bins, patches = plt.hist(f, 'auto', facecolor='blue', alpha=0.5)
        
        plt.title(title)
        plt.show()

def standardize_pixels(pixels):
    mean, std = pixels.mean(), pixels.std()
    pixels = (pixels - mean) / std
    # confirm it had the desired effect
    mean, std = pixels.mean(), pixels.std()
    print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    return pixels

class fiducial_dataset:
    
    def __init__(self, path):
        # load the data
        print('Load data from:', path)
        
        with h5py.File(path, 'r') as hf:
            x_train_raw = hf["X_train"][:]
            x_test_raw = hf["X_test"][:]
            x_predict_raw = hf["X_predict"][:]
            x_predict_noise_raw = hf["X_predict_noise"][:]
            x_rhino_raw = hf["X_rhino"][:]
            x_screw_raw = hf["X_screw"][:]
            x_noise_raw = hf["X_noise"][:]
            y_train_raw = hf["y_train"][:]
            y_test_raw = hf["y_test"][:]
            y_predict_raw = hf["y_predict"][:]
            y_predict_noise_raw = hf["y_predict_noise"][:]
            y_rhino_raw = hf["y_rhino"][:]
            y_screw_raw = hf["y_screw"][:]
            y_noise_raw = hf["y_noise"][:]
         
            
# data normalization was tested, but it did not reflect better results
#        x_train_raw = 255. - x_train_raw
#        x_test_raw = 255. - x_test_raw
#        x_predict_raw = 255. - x_predict_raw
#        x_predict_noise_raw = 255. - x_predict_noise_raw
#        x_rhino_raw = 255. - x_rhino_raw
#        x_screw_raw = 255. - x_screw_raw
#        x_noise_raw = 255. - x_noise_raw
#        
##        print(x_predict_raw)
#
#        x_train_raw /= 255.0       
#        x_test_raw /= 255.0 
#        x_predict_raw /= 255.0
#        x_predict_noise_raw /= 255.0
#        x_rhino_raw /= 255.0
#        x_screw_raw /= 255.0
#        x_noise_raw /= 255.0
        
#        x_train_raw = standardize_pixels(x_train_raw)
#        x_test_raw = standardize_pixels(x_test_raw)
#        x_predict_raw = standardize_pixels(x_predict_raw)
#        x_predict_noise_raw = standardize_pixels(x_predict_noise_raw)
#        x_rhino_raw = standardize_pixels(x_rhino_raw)
#        x_screw_raw = standardize_pixels(x_screw_raw)
#        x_noise_raw = standardize_pixels(x_noise_raw)
#                         
#        x_train_raw=x_train_raw*(1./255.)
#        x_test_raw=x_test_raw*(1./255.)
#        x_predict_raw=x_predict_raw*(1./255.)
#        x_predict_noise_raw=x_predict_noise_raw*(1./255.)
#        x_rhino_raw=x_rhino_raw*(1./255.)
#        x_screw_raw=x_screw_raw*(1./255.)
#        x_noise_raw=x_noise_raw*(1./255.)

        #print(x_predict_raw)

        #histogram_intensity(x_screw_raw, 'screw')
        #histogram_intensity(x_rhino_raw, 'rhino')
        #histogram_intensity(x_noise_raw, 'noise')

        #x_train_raw = x_train_raw[:1000] 
        #y_train_raw = y_train_raw[:1000]
         
        self.num_classes = 2
        self.entropic_entropy = 1 / self.num_classes

        
        # NORMAL DATA
        ######################################################       
        self.X_noise = data_transform(x_noise_raw)  
        self.X_rhino = data_transform(x_rhino_raw)
        self.X_screw = data_transform(x_screw_raw)
        self.X_predict = data_transform(x_predict_raw)
        self.X_predict_noise = data_transform(x_predict_noise_raw)
        
        self.Y_noise = y_noise_raw
        self.Y_rhino = y_rhino_raw
        self.Y_screw = y_screw_raw     
        self.Y_predict = y_predict_raw  
        self.Y_predict_noise = y_predict_noise_raw         
        
        # TRAIN  & VAL
        ######################################################       
        self.X_train = data_transform(x_train_raw)
        self.X_val = data_transform(x_test_raw)    
            
        self.Y_train = to_categorical(y_train_raw, self.num_classes)
        self.Y_val = to_categorical(y_test_raw, self.num_classes)  
        self.Y_val_raw = y_test_raw
     
        # PREDICTION 
        ######################################################

