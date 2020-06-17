import os
import numpy as np

from matplotlib import pyplot

from keras.backend.tensorflow_backend import set_session
from keras.optimizers import SGD, Adam, Adamax
import tensorflow as tf

import datetime

gpu_name = '0'

model_id = 0

#classifier_name = 'softmax'
#classifier_name = 'background'
#classifier_name = 'cross_loss'
classifier_name = 'ring_loss'


softmax = True
BG = False
entropic_openset = False
objectosphere = False


cross_entropy_loss_weight = 1.0
ring_loss_weight = 0.001
Minimum_Knowns_Magnitude = 20.
optimizer_name = 'adamx'

lr = 0.001
batch_size = 90
epochs = 1000
epochs_patience = 70 #30
results_dir = 'Model_Results/Models/'

data_path = 'fiducial_dataset.h5'

"""
Setting GPU to use.
"""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = gpu_name
set_session(tf.Session(config=config))

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
if optimizer_name == 'adam':
    optimizer = Adam(lr=lr)
elif optimizer_name == 'adamx':
    optimizer = Adamax(learning_rate=lr, beta_1=0.9, beta_2=0.999)
else:
    optimizer = SGD(lr=lr)

now = datetime.datetime.now()
print ("Training started at: ")
print (now.strftime("%Y-%m-%d %H:%M:%S"))

if classifier_name=='softmax':
    import training_softmax
    training_softmax.softmax_train(results_dir, model_id, batch_size, epochs, optimizer, data_path, epochs_patience)

elif classifier_name=='background':
    import training_background
    training_background.BG_train(results_dir, model_id, batch_size, epochs, optimizer, data_path, epochs_patience)
    
elif classifier_name=='cross_loss':
    import training_cross
    training_cross.cross_train(results_dir, model_id, batch_size, epochs, optimizer, data_path, epochs_patience)

elif classifier_name=='ring_loss':
    import training_objectosphere
    training_objectosphere.objectosphere_train(results_dir, model_id, batch_size, epochs, optimizer, data_path, epochs_patience, Minimum_Knowns_Magnitude, cross_entropy_loss_weight, ring_loss_weight)

else:
    print('Unknown classifier selected')

now = datetime.datetime.now()
print ("Training ended at: ")
print (now.strftime("%Y-%m-%d %H:%M:%S"))