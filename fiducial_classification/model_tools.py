from keras.models import Sequential
from keras.layers import Dense, Activation,Flatten,Input,Conv3D,MaxPooling3D,Dropout,BatchNormalization,Concatenate
from keras.models import Model
from keras import backend as K
import numpy as np

num_classes = 2

def LeNet_Extended_3D(perform_L2_norm=False,activation_type='softmax',ring_approach=False,background_class=False, knownsMinimumMag = None):
    """
    Defines the network architecture for LeNet++.
    Use the options for different approaches:
    background_class: Classification with additional class for negative classes
    ring_approach: ObjectoSphere Loss applied if True
    knownsMinimumMag: Minimum Magnitude allowed for samples belonging to one of the Known Classes if ring_approach is True
    """
    image = Input(shape=(38, 38, 38, 1), dtype='float32', name='image')

    #conv1_1 = Conv3D(16, (3,3,3), strides=1, padding="same",name='conv1_3')(mnist_image)
    #conv1_2 = Conv3D(16, (3,3,3), strides=1, padding="same",name='conv1_4')(conv1_1)
    #conv1_2 = BatchNormalization(name='BatchNormalization_12')(conv1_2)
    
    # 38 X 38 X 38 --> 19 X 19 X 19
    conv1_1 = Conv3D(32, (5,5,5), strides=1, padding="same",name='conv1_1')(image)
    conv1_2 = Conv3D(32, (5,5,5), strides=1, padding="same",name='conv1_2')(conv1_1)
    conv1_2 = BatchNormalization(name='BatchNormalization_1')(conv1_2)  
    pool1 = MaxPooling3D(pool_size=(2,2,2), strides=2,name='pool1')(conv1_2)
    
    # 19 X 19 X 19 --> 9 X 9 X 9
    conv2_1 = Conv3D(64, (3,3,3), strides=1, padding="same", name='conv2_1')(pool1)
    conv2_2 = Conv3D(64, (3,3,3), strides=1, padding="same", name='conv2_2')(conv2_1)
    conv2_2 = BatchNormalization(name='BatchNormalization_2')(conv2_2)
    pool2 = MaxPooling3D(pool_size=(2,2,2), strides=2, name='pool2')(conv2_2)
    
    # 9 X 9 X 9 --> 4 X 4 X 4
    conv3_1 = Conv3D(128, (2,2,2), strides=1, padding="same",name='conv3_1')(pool2)
    conv3_2 = Conv3D(128, (2,2,2), strides=1, padding="same",name='conv3_2')(conv3_1)
    conv3_2 = BatchNormalization(name='BatchNormalization_3')(conv3_2)
    pool3 = MaxPooling3D(pool_size=(2,2,2), strides=2, name='pool3')(conv3_2)
    
    #pool3 = Dropout(name='dropout', rate=0.2)(pool3)
    flatten=Flatten(name='flatten')(pool3)
    fc = flatten
    #fc = Dense(512,name='fc1',use_bias=True)(flatten)
    #fc = Dense(512,name='fc2',use_bias=True)(fc)
    #fc = Dense(256,name='fc',use_bias=True)(fc)
    #fc = Dropout(name='dropout', rate=0.2)(fc)
    fc = Dense(2,name='fc',use_bias=True)(fc)

    if perform_L2_norm:
        alpha_multipliers = Input((1,), dtype='float32', name='alphas')
        act = Activation(lambda x: alpha_multipliers*(K.l2_normalize(x,axis=1)),name='act')(fc)
        pred = Dense(num_classes, activation=activation_type,name='pred',use_bias=False)(act)
        model = Model(inputs=[image,alpha_multipliers], outputs=[pred])
    elif knownsMinimumMag is not None:
        knownUnknownsFlag = Input((1,), dtype='float32', name='knownUnknownsFlag')
        pred = Dense(num_classes, name='pred',use_bias=False)(fc)
        softmax = Activation(activation_type,name='softmax')(pred)
        model = Model(inputs=[image,knownsMinimumMag], outputs=[softmax,fc])
    elif background_class:
        pred = Dense(num_classes+1, name='pred',use_bias=False)(fc)
        softmax = Activation(activation_type,name='softmax')(pred)
        model = Model(inputs=[image], outputs=[softmax])
    else:
        pred = Dense(num_classes, name='pred', use_bias=False)(fc)
        softmax = Activation(activation_type,name='softmax')(pred)
        model = Model(inputs=[image], outputs=[softmax])
    return model

def extract_features(model,data,layer_name = ['fc','softmax']):
    """
    Use this function to extract deep feature from the layers of interest.
    Input:
        model: Keras model object
        data: The data in Numpy array for which deep features need to be extracted.
        layer_name: A list containing all the layer names that need to be extracted.
    """
    print('shape model:', model.input_shape)
    out=[]
    for l in layer_name:
        out.append(model.get_layer(l).output)
    intermediate_layer_model = Model(inputs=model.input,outputs=out)
    if len(model.input_shape)==5: # MR: added for 3D CNN
        return intermediate_layer_model.predict([data])
    #elif len(model.input_shape)==4:
    #    intermediate_output = intermediate_layer_model.predict([data])
    #elif len(model.input_shape)==3:
    #    intermediate_output = intermediate_layer_model.predict([data,np.ones(data.shape[0]),np.ones(data.shape[0])])        
    #else:
    #    intermediate_output = intermediate_layer_model.predict([data,np.ones(data.shape[0])])
    #return intermediate_output

def concatenate_training_data_3D(obj,y,cross_entropy_probs,ring_loss=False):
    """
    Parameters:
        obj: is an object from a class in file data_prep.py
        y: are the images from the class that needs to be trained as negatives example cifar.images or letters.images
        cross_entropy_probs: Multiplier to the categorical labels
        ring_loss: Boolean value returns Y_pred_flags (Default:False)
    Returns:
        X_train_data: Numpy array containing training samples
        Y_train_data: Numpy array containing Label values
        sample_weights: 1D Numpy array containing weight of each sample
        Y_train_flags: Returned only when ring_loss=True. Numpy array containing flags indicating the sample is a known versus known unknown
    """
    X_train_data=np.concatenate((obj.X_train,y))
    Y_train_data=np.concatenate((obj.Y_train,np.ones((y.shape[0],num_classes))*cross_entropy_probs)) # 10 to 2
    class_no=np.argmax(obj.Y_train,axis=1)
    sample_weights = []
    sample_weights_knowns=np.zeros_like(class_no).astype(np.float32)
    for cls in range(obj.Y_train.shape[1]):
        sample_weights_knowns[class_no==cls]=100./len(class_no[class_no==cls])
    sample_weights=np.concatenate([sample_weights_knowns,np.ones((y.shape[0]))*(100./y.shape[0])])
    if ring_loss:
        Y_train_flags=np.zeros((X_train_data.shape[0],2)) # 
        Y_train_flags[:obj.X_train.shape[0],0]=1
        Y_train_flags[obj.X_train.shape[0]:,1]=1
        return X_train_data,Y_train_data,sample_weights,Y_train_flags
    else:
        return X_train_data,Y_train_data,sample_weights

