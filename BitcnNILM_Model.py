from Arguments import *
from Logger import log
from keras.models import Input, Model
from keras.layers import Dense, Conv2D, Flatten, Reshape, Lambda, dot, Activation, concatenate, Conv1D, SpatialDropout1D, BatchNormalization, add  
from keras.utils import print_summary, plot_model
import numpy as np
import keras.backend as K
import os
from typing import List, Tuple
import keras.backend as K
#import keras.layers
from keras import optimizers
from keras.engine.topology import Layer

#######################
# import tensorflow as tf # if tensorflow 1

import tensorflow.compat.v1 as tf # if using tensorflow 2
tf.disable_v2_behavior()
########################
import h5py
import argparse


# Model setting begin, used in Sequence to point Learning based on bidirectional dilated residual network for nilm  
nb_filters=128
filter_length = 3
dilations=[1,2,4,8,16,32,64,128]
dropout = 0.3


def residual_block(x, dilation_rate, nb_filters, kernel_size, padding, activation='relu', dropout_rate=0,
                   kernel_initializer='he_normal'):
    prev_x = x
    for k in range(2):
        x = Conv1D(filters=nb_filters,
                   kernel_size=kernel_size,
                   dilation_rate=dilation_rate,
                   kernel_initializer=kernel_initializer,
                   padding=padding)(x)
        x = BatchNormalization()(x)  
        x = Activation('relu')(x)
        x = SpatialDropout1D(rate=dropout_rate)(inputs=x)

    # 1x1 conv to match the shapes (channel dimension).
    prev_x = Conv1D(nb_filters, 1, padding='same')(prev_x)
    # 
    res_x = add([prev_x, x])
    #####################################
    res_x = Activation(activation)(res_x)
    return res_x, x

def get_model(appliance, input_tensor, window_length, transfer_dense=False, transfer_cnn=False,
              cnn='kettle', pretrainedmodel_dir='./models/'):

    reshape = Reshape((window_length, 1),)(input_tensor)
    x = reshape
    x = Conv1D(filters=nb_filters, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    skip_connections = []
    for d in dilations:
        x, skip_out = residual_block(x, dilation_rate=d,nb_filters=nb_filters,kernel_size=filter_length, padding = 'same', activation = 'relu', dropout_rate=dropout)
        skip_connections.append(skip_out)
    x = add(skip_connections)
    x = Lambda(lambda t: t[:, -1, :])(x)
    d_out = Dense(1, activation='linear', name='output')(x) 

    model = Model(inputs=input_tensor, outputs=d_out)

# Model setting done


####model structure done!


##############################
    #session = K.get_session() # For Tensorflow 1
    session = tf.keras.backend.get_session() #For Tensorflow 2


##############################

    # For transfer learning
    if transfer_dense:
        log("Transfer learning...")
        log("...loading an entire pre-trained model")
        weights_loader(model, pretrainedmodel_dir+'/cnn_s2p_' + appliance + '_pointnet_model')
        model_def = model
    elif transfer_cnn and not transfer_dense:
        log("Transfer learning...")
        log('...loading a ' + appliance + ' pre-trained-cnn')
        cnn_weights_loader(model, cnn, pretrainedmodel_dir)
        model_def = model
        for idx, layer1 in enumerate(model_def.layers):
            if hasattr(layer1, 'kernel_initializer') and 'conv2d' not in layer1.name and 'cnn' not in layer1.name:
                log('Re-initialize: {}'.format(layer1.name))
                layer1.kernel.initializer.run(session=session)

    elif not transfer_dense and not transfer_cnn:
        log("Standard training...")
        log("...creating a new model.")
        model_def = model
    else:
        raise argparse.ArgumentTypeError('Model selection error.')

    # Printing, logging and plotting the model
    #print_summary(model_def)
    plot_model(model, to_file='./model.png', show_shapes=True, show_layer_names=True, rankdir='TB')

    # Adding network structure to both the log file and output terminal
    files = [x for x in os.listdir('./') if x.endswith(".log")]
    with open(max(files, key=os.path.getctime), 'a') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model_def.summary(print_fn=lambda x: fh.write(x + '\n'))
    return model_def

    # Check weights slice
    # for v in tf.trainable_variables():
    #     if v.name == 'conv2d_1/kernel:0':
    #         cnn1_weights = session.run(v)
    # return model_def, cnn1_weights


def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))


def cnn_weights_loader(model_to_fill, cnn_appliance, pretrainedmodel_dir):
    log('Loading cnn weights from ' + cnn_appliance)    
    weights_path = pretrainedmodel_dir+'/cnn_s2p_' + cnn_appliance + '_pointnet_model' + '_weights.h5'
    if not os.path.exists(weights_path):
        print('The directory does not exist or you do not have the files for trained model')
        
    f = h5py.File(weights_path, 'r')
    log(f.visititems(print_attrs))
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    for name in layer_names:
        if 'conv2d_' in name or 'cnn' in name:
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in weight_names]

            model_to_fill.layers[int(name[-1])+1].set_weights(weight_values)
            log('Loaded cnn layer: {}'.format(name))

    f.close()
    print('Model loaded.')


def weights_loader(model, path):
    log('Loading cnn weights from ' + path)
    model.load_weights(path + '_weights.h5')





