"""
Created on Thu 2020

####################################################################################
# This code is to train a neural network to perform energy disaggregation,
# i.e., given a sequence of electricity mains reading, the algorithm
# separates the mains into appliances.
#
# Inputs: mains windows -- find the window length in params_appliance
# Targets: appliances windows --
#
#
# This code is written by Jia, Ziyue based on the code from
#  1. https://github.com/cbrewitt/nilm_fcn (which was written by Cillian Brewitt)
#  2. https://github.com/MingjunZhong/NeuralNetNilm (which was written by Chaoyun Zhang and Mingjun Zhong)

# References:

1. Brewitt, Cillian , and N. Goddard . "Non-Intrusive Load Monitoring with Fully
Convolutional Networks." (2018). arXiv:1812.03915


2.  Chaoyun Zhang, Mingjun Zhong, Zongzuo Wang, Nigel Goddard, and Charles Sutton.
# ``Sequence-to-point learning with neural networks for nonintrusive load monitoring."
# Thirty-Second AAAI Conference on Articial Intelligence (AAAI-18), Feb. 2-7, 2018.
####################################################################################

Updated on 2020:
- Python 3

"""
from Arguments import *
from Logger import log
from keras.models import Model
from keras.layers import Dense, Conv1D, Flatten, Reshape, Lambda
from keras.utils import print_summary, plot_model
import numpy as np
import keras.backend as K
import os
#######################
# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
########################
import h5py
import argparse


def get_model(appliance, input_tensor, input_window_length, transfer_dense=False, transfer_cnn=False,
              cnn='kettle', n_dense=1, pretrainedmodel_dir='./models/'):
    if n_dense == 1:
        print(" ")

    output_length = 2053
    offset = (input_window_length - output_length)//2


    reshape = Reshape((input_window_length, 1),
                      )(input_tensor)
    # CNN
    x = Conv1D(128, 9, padding='same', activation='relu', dilation_rate=1)(reshape)
    # dilated CNN
    x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=2)(x)
    x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=4)(x)
    x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=8)(x)
    x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=16)(x)
    x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=32)(x)
    x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=64)(x)
    x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=128)(x)
    x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=256)(x)
    # CNN
    x = Conv1D(256, 1, padding='same', activation='relu')(x)
    x = Conv1D(1, 1, padding='same', activation=None)(x)

    x = Reshape((input_window_length,), input_shape=(input_window_length, 1))(x)
    x = Lambda(lambda x: x[:, offset:-offset], output_shape=(output_length,))(x)

    model = Model(inputs=input_tensor, outputs=x)
##############################
    #session = K.get_session()
    session = tf.keras.backend.get_session()


##############################

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
    print_summary(model_def)
    # plot_model(model, to_file='./model.png', show_shapes=True, show_layer_names=True, rankdir='TB')

    # Adding network structure to both the log file and output terminal
    files = [x for x in os.listdir('./') if x.endswith(".log")]
    with open(max(files, key=os.path.getctime), 'a') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model_def.summary(print_fn=lambda x: fh.write(x + '\n'))

    # # Check weights slice
    # for v in tf.trainable_variables():
    #     if v.name == 'conv2d_1/kernel:0':
    #         cnn1_weights = session.run(v)
    return model_def


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





