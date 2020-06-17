import numpy as np

from matplotlib import pyplot

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from keras import backend as K

import fiducial_dataset
import model_tools

def cross_train(results_dir, model_id, batch_size, epochs, optimizer, data_path, epochs_patience):
    
    fiducial = fiducial_dataset.fiducial_dataset(data_path)

    X_train,Y_train,sample_weights=model_tools.concatenate_training_data_3D(fiducial,fiducial.X_noise,fiducial.entropic_entropy)
    
    print(Y_train[0])    
    print(Y_train[len(Y_train)-55])
    
    save_to_file = results_dir+'Cross_'+str(model_id)+'.h5py'
    print('save model to file:', save_to_file)
    
    model_saver = ModelCheckpoint(
            save_to_file,
            monitor='val_loss',
            verbose=0, 
            save_best_only=True, 
            save_weights_only=False, 
            mode='min', 
            period=1)
    
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=epochs_patience)

    callbacks_list = [model_saver, early_stopping]
    
    model=model_tools.LeNet_Extended_3D()

    model.compile(optimizer=optimizer,loss={'softmax': 'categorical_crossentropy'},metrics=['accuracy'])
    
    history=model.fit(
            x=[X_train],
            y=[Y_train],
            validation_data=[fiducial.X_val,fiducial.Y_val],
            batch_size=batch_size,
            epochs=epochs,verbose=1,
            callbacks=callbacks_list,
            sample_weight=sample_weights)
    
    # Plot learning curves
    approach = 'Entropic OpenSet'
    lrate = K.eval(model.optimizer.lr)
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.title(approach + ' lrate='+str(lrate), pad=-50)
    pyplot.show()
