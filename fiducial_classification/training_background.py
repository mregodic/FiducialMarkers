import numpy as np

from matplotlib import pyplot

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import fiducial_dataset
import model_tools

def BG_train(results_dir, model_id, batch_size, epochs, optimizer, data_path, epochs_patience):
    
    fiducial = fiducial_dataset.fiducial_dataset(data_path)

    model_saver = ModelCheckpoint(
            results_dir+'BG_'+str(model_id)+'.h5py', 
            monitor='val_loss', 
            verbose=0, 
            save_best_only=True, 
            save_weights_only=False, 
            mode='min', 
            period=1)
    
    early_stopping = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=epochs_patience)

    callbacks_list = [model_saver, early_stopping]

    X_train=np.concatenate((fiducial.X_train,fiducial.X_noise))
    tmp_mnist = np.append(fiducial.Y_train,np.zeros((fiducial.Y_train.shape[0],1)),1)
    tmp_neg = np.zeros((fiducial.X_noise.shape[0],fiducial.num_classes+1))
    tmp_neg[:,-1]=1
    Y_train=np.concatenate((tmp_mnist,tmp_neg))

    model=model_tools.LeNet_Extended_3D(background_class=True)
    
    model.compile(
            optimizer=optimizer,
            loss={'softmax': 'categorical_crossentropy'},
            metrics=['categorical_accuracy'])
    
    history=model.fit(
            x=[X_train],
            y=[Y_train],
            validation_data=[fiducial.X_val,np.append(fiducial.Y_val,np.zeros((fiducial.Y_val.shape[0],1)),1)],
            batch_size=batch_size,
            epochs=epochs,
            verbose=1, 
            callbacks=callbacks_list)
    
    approach = 'Background'
    lrate = K.eval(model.optimizer.lr)
    pyplot.plot(history.history['categorical_accuracy'], label='train')
    pyplot.plot(history.history['val_categorical_accuracy'], label='test')
    pyplot.title(approach + ' lrate='+str(lrate), pad=-50)
    pyplot.show()