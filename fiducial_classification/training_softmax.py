import numpy as np

from matplotlib import pyplot

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import fiducial_dataset
import model_tools

def softmax_train(results_dir, model_id, batch_size, epochs, optimizer, data_path, epochs_patience):
    
    fiducial = fiducial_dataset.fiducial_dataset(data_path)
    
    model_saver = ModelCheckpoint(
                                    results_dir+'SoftMax_'+str(model_id)+'.h5py', 
                                    monitor='val_loss', 
                                    verbose=0, 
                                    save_best_only=True,
                                    save_weights_only=False, 
                                    mode='min', 
                                    period=1
                                )
    
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=epochs_patience)
    
    callbacks_list = [model_saver, early_stopping]
    
    model = model_tools.LeNet_Extended_3D()

    model.compile(
            optimizer=optimizer, 
            loss={'softmax': 'categorical_crossentropy'}, 
            metrics=['accuracy'])
    
    history=model.fit(
            x=[fiducial.X_train],
            y=[fiducial.Y_train],
            validation_data=[fiducial.X_val,fiducial.Y_val],
            batch_size=batch_size, epochs=epochs,verbose=1, callbacks=callbacks_list)
    
    # Plot learning curves
    approach = 'SoftMax'
    lrate = K.eval(model.optimizer.lr)
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.title(approach + ' lrate='+str(lrate), pad=-50)
    pyplot.show()
