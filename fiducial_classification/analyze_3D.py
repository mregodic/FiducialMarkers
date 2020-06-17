# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 19:21:49 2020

@author: Milovan
"""
import keras.backend as Kb


import model_tools
import visualizing_tools
import evaluation_tools
import fiducial_dataset


import os
import numpy as np
import pandas as pd
import keras
from keras import backend as K
from matplotlib import pyplot as plt



GPU_NO="0"
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = GPU_NO
set_session(tf.Session(config=config))

if not os.path.exists('Model_Results/Final_Plots'):
    os.makedirs('Model_Results/Final_Plots')
if not os.path.exists('Model_Results/DIRs'):
    os.makedirs('Model_Results/DIRs')
    
fiducial = fiducial_dataset.fiducial_dataset('fiducial_dataset.h5')


def analyze_3D(model,         
            pos_x=fiducial.X_predict,  # X_predict
            pos_y=fiducial.Y_predict, # Y_predict               
            neg=fiducial.X_predict_noise, # X_predict_noise
            neg_labels='Not_Fiducial',
            file_name=''):
    
    print('number of predict:', len(pos_y))
    print('number of noise:', len(neg))
    
    print('len x', len(pos_x))
    print('len y', len(pos_y))
    print('len noise', len(neg))
  
    intermediate_output=model_tools.extract_features(model,pos_x,layer_name=['fc','softmax','pred'])
    if neg is not None:
        neg_intermediate_output=model_tools.extract_features(model,neg,layer_name=['fc','softmax','pred'])
    pred_weights=model.get_layer('pred').get_weights()[0]
    

    visualizing_tools.plotter_2D(
                                    intermediate_output[0],
                                    pos_y,
                                    neg_intermediate_output[0],
                                    final=True,
                                    file_name='Model_Results/Final_Plots/'+file_name,
                                    pos_labels='Unknown Structures',
                                    neg_labels=neg_labels,
                                    pred_weights=pred_weights
                                )
#    
    visualizing_tools.plot_softmax_histogram(
                                                intermediate_output[1],
                                                neg_intermediate_output[1],
                                                file_name='Model_Results/Final_Plots/'+file_name,
                                                pos_labels='Unknown Structures',
                                                neg_labels=neg_labels
                                            )

    gt_y = np.concatenate((pos_y,np.ones(neg_intermediate_output[1].shape[0])*fiducial.num_classes),axis=0)
    
    pred_y = np.concatenate((intermediate_output[1],neg_intermediate_output[1]),axis=0)
    
    evaluation_tools.write_file_for_DIR(gt_y,
                                        pred_y,
                                        file_name=('Model_Results/DIRs/'+file_name).format(neg_labels,'txt'),
                                        num_of_known_classes=fiducial.num_classes
                                       )
    evaluation_tools.write_file_for_DIR(gt_y,
                                        pred_y,
                                        file_name=('Model_Results/DIRs/'+file_name).format(neg_labels,'txt'),
                                        feature_vector=np.concatenate((intermediate_output[0],neg_intermediate_output[0])),
                                        num_of_known_classes=fiducial.num_classes
                                       )

strMultiplying = 'Multiplying_with_mag_'
#strMultiplying = ''

def evaluation_plotter(dataset_type,random_model_no='0', known_labels=[], negative_label = fiducial.num_classes):
    evaluation_tools.process_files(DIR_filename='Model_Results/Final_Plots/'+dataset_type+'/DIR_Unknowns_'+random_model_no,
                                   files_to_process=[
                                                        'Model_Results/DIRs/'+dataset_type+'/' + 'SoftMax_'+random_model_no+'_'+dataset_type+'.txt',
                                                        'Model_Results/DIRs/'+dataset_type+'/' + 'BG_'+random_model_no+'_'+dataset_type+'.txt',
                                                        'Model_Results/DIRs/'+dataset_type+'/' + 'Cross_'+random_model_no+'_'+dataset_type+'.txt',
                                                        'Model_Results/DIRs/'+dataset_type+'/' + strMultiplying + 'Ring_'+str(Minimum_mag_for_knowns)+'_'+random_model_no+'_'+dataset_type+'.txt'
                                                    ],
                                   labels=['Softmax Thresholding','Background Class','Entropic Openset','Objectosphere'],
                                   out_of_plot=False,
                                   known_labels=known_labels,
                                   negative_label=negative_label
                                )





random_model_no='0'
dataset_type='Not_Fiducial'
if not os.path.exists('Model_Results/Final_Plots/'+dataset_type):
    os.makedirs('Model_Results/Final_Plots/'+dataset_type)
if not os.path.exists('Model_Results/DIRs/'+dataset_type):
    os.makedirs('Model_Results/DIRs/'+dataset_type)
    
print(fiducial.Y_predict[0])

def softmax_analyze():
    model=keras.models.load_model('Model_Results/Models/SoftMax_'+random_model_no+'.h5py')
    analyze_3D(model,
            file_name=dataset_type+'/SoftMax_'+random_model_no+'_{}.{}')

def BG_analyze():
    model=keras.models.load_model('Model_Results/Models/BG_'+random_model_no+'.h5py')
    analyze_3D(model,file_name=dataset_type+'/BG_'+random_model_no+'_{}.{}')

def cross_analyze():
    model=keras.models.load_model('Model_Results/Models/Cross_'+random_model_no+'.h5py')
    analyze_3D(model,
            neg_labels='Not_Fiducial',
            file_name=dataset_type+'/Cross_'+random_model_no+'_{}.{}', )
        
from keras.layers import Input
Minimum_mag_for_knowns=30.


def ring_loss(y_true,y_pred):
    pred=K.sqrt(K.sum(K.square(y_pred),axis=1))
    error=K.mean(K.square(
        # Loss for Knowns having magnitude greater than knownsMinimumMag
        y_true[:,0]*(K.maximum(knownsMinimumMag-pred,0.))
        # Add two losses
        +
        # Loss for unKnowns having magnitude greater than unknownsMaximumMag
        y_true[:,1]*(pred)
    ))
    return error

knownsMinimumMag = Input((1,), dtype='float32', name='knownsMinimumMag')

def ring_analyze():
    
    model=keras.models.load_model(('Model_Results/Models/Ring_{}_{}.h5py').format(Minimum_mag_for_knowns,random_model_no), custom_objects={'ring_loss': ring_loss})
    
    analyze_3D(model,
            file_name=dataset_type+'/Ring_'+str(Minimum_mag_for_knowns)+'_'+random_model_no+'_{}.{}', )


softmax_analyze()
BG_analyze()
cross_analyze()
ring_analyze()
    
# MR: single_class_eval is a parameter that specifies to evaluate only one class versus unknowns [known_class, unknown_class]
#known_labels = [0] # screws
known_labels = [1] # spherical
#known_labels = [0, 1] # screw and spherical combined

negative_label = fiducial.num_classes
#single_class_eval = [1, 2] # spherical fiducials
#single_class_eval = [] # all

evaluation_plotter(dataset_type,'0', known_labels, negative_label)
