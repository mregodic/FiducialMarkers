import numpy as np

from matplotlib import pyplot

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras import backend as K

import fiducial_dataset
import model_tools


knownsMinimumMag = 0

"""
Objectosphere loss function.
"""
def ring_loss(y_true,y_pred):
    pred=K.sqrt(K.sum(K.square(y_pred),axis=1))
    error=K.mean(K.square(
        # Loss for Knowns having magnitude greater than knownsMinimumMag
        y_true[:,0]*(K.maximum(knownsMinimumMag-pred,0.))
        # Add two losses
        +
        # Loss for unKnowns having magnitude greater than unknownsMaximumMag
        y_true[:,1]*pred
    ))
    return error

def objectosphere_train(results_dir, model_id, batch_size, epochs, optimizer, data_path, epochs_patience, Minimum_Knowns_Magnitude, cross_entropy_loss_weight, ring_loss_weight):
    
    
    fiducial = fiducial_dataset.fiducial_dataset(data_path)

    X_train,Y_train,sample_weights,Y_pred_with_flags=model_tools.concatenate_training_data_3D(fiducial,fiducial.X_noise,fiducial.entropic_entropy,ring_loss=True)
    knownsMinimumMag = Input((1,), dtype='float32', name='knownsMinimumMag')
    knownsMinimumMag_ = np.ones((X_train.shape[0]))*Minimum_Knowns_Magnitude
    
    print('knownsMinimumMag=', knownsMinimumMag_)

    # debugging prints
    #print('entropic scale=', fiducial.entropic_entropy)
    #print(Y_train[len(Y_train)-55])
    #print('shape x train[0]=', X_train.shape[0])
    
    model=model_tools.LeNet_Extended_3D(ring_approach=True,knownsMinimumMag=knownsMinimumMag)
        
    model_saver = ModelCheckpoint(
            results_dir+'Ring_'+str(Minimum_Knowns_Magnitude)+'_'+str(model_id)+'.h5py',
            monitor='val_loss', 
            verbose=0, 
            save_best_only=True,
            save_weights_only=False,
            mode='min', 
            period=1)
    
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=epochs_patience)

    callbacks_list = [model_saver, early_stopping]
    
    flag_placeholder=np.zeros((fiducial.Y_val.shape[0],2))
    flag_placeholder[:,0]=1

    model.compile(
            optimizer=optimizer,
            loss={'softmax': 'categorical_crossentropy','fc':ring_loss},
            loss_weights={'softmax': cross_entropy_loss_weight, 'fc': ring_loss_weight},
            metrics=['accuracy'])
    
    history=model.fit(
            x=[X_train,knownsMinimumMag_],
            y=[Y_train,Y_pred_with_flags],
            validation_data=[
                    [fiducial.X_val,np.ones(fiducial.X_val.shape[0])*Minimum_Knowns_Magnitude],
                    [fiducial.Y_val,flag_placeholder]],
            batch_size=batch_size, 
            epochs=epochs,
            verbose=1,
            sample_weight=[sample_weights,sample_weights],
            callbacks=callbacks_list)
    
    approach = 'Objectosphere'
    lrate = K.eval(model.optimizer.lr)
    pyplot.plot(history.history['softmax_accuracy'], label='softmax train')
    pyplot.plot(history.history['val_softmax_accuracy'], label='softmax test')
    pyplot.plot(history.history['fc_accuracy'], label='objectosphere train')
    pyplot.plot(history.history['val_fc_accuracy'], label='objectosphere test')
    pyplot.title(approach + ' m=' + str(Minimum_Knowns_Magnitude) + ' lrate='+str(lrate), pad=-50)
    pyplot.show()

