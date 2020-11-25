"""
reproduce by Jia,ziyue

"""

import os
from fcn_Model import get_model
from DataProvider import ChunkDoubleSourceSlider2, ChunkS2S_Slider_fcn
import NetFlowExt as nf
from Logger import log
#####original tensorflow 1

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#############
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.layers import Input
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from Arguments import *

def remove_space(string):
    return string.replace(" ","")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network\
                                     for energy disaggregation - \
                                     network input = mains window; \
                                     network target = the states of \
                                     the target appliance.')
    parser.add_argument('--appliance_name',
                        type=remove_space,
                        default='kettle',
                        help='the name of target appliance')
    parser.add_argument('--datadir',
                        type=str,
                        default='/media/michele/Dati/myREFIT/',
                        help='this is the directory of the training samples')
    parser.add_argument('--pretrainedmodel_dir',
                        type=str,
                        default='./pretrained_model',
                        help='this is the directory of the pre-trained models')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./models',
                        help='this is the directory to save the trained models')
    parser.add_argument('--batchsize',
                        type=int,
                        default=256,
                        help='The batch size of training examples')
    parser.add_argument('--n_epoch',
                        type=int,
                        default=300,
                        help='The number of epochs.')
    parser.add_argument('--save_model',
                        type=int,
                        default=-1,
                        help='Save the learnt model:\
                        0 -- not to save the learnt model parameters;\
                        n (n>0) -- to save the model params every n steps;\
                        -1 -- only save the learnt model params\
                        at the end of training.')
    parser.add_argument('--dense_layers',
                        type=int,
                        default=1,
                        help=':\
                                1 -- One dense layers (default Seq2point);\
                                2 -- Two dense layers;\
                                3 -- Three dense layers.')
    parser.add_argument("--transfer_model", type=str2bool,
                        default=False,
                        help="True: using entire pre-trained model.\
                             False: retrain the entire pre-trained model;\
                             This will override the 'transfer_cnn' and 'cnn' parameters;\
                             The appliance_name parameter will use to retrieve \
                             the entire pre-trained model of that appliance.")
    parser.add_argument("--transfer_cnn", type=str2bool,
                        default=False,
                        help="True: using a pre-trained CNN\
                              False: not using a pre-trained CNN.")
    parser.add_argument('--cnn',
                        type=str,
                        default='kettle',
                        help='The CNN trained by which appliance to load (pretrained model).')
    parser.add_argument('--gpus',
                        type=int,
                        default=-1,
                        help='Number of GPUs to use:\
                            n -- number of GPUs the system should use;\
                            -1 -- do not use any GPU.')
    parser.add_argument('--crop_dataset',
                        type=int,
                        default=None,
                        help='for debugging porpose should be helpful to crop the training dataset size')
    parser.add_argument('--ram',
                        type=int,
                        default=5*10**5,
                        help='Maximum number of rows of csv dataset can handle without loading in chunks')
    return parser.parse_args()


args = get_arguments()
log('Arguments: ')
log(args)

# some constant parameters
CHUNK_SIZE = 5*10**6

# start the session for training a network
sess = tf.InteractiveSession()

# the appliance to train on
appliance_name = args.appliance_name

# path for training data
training_path = args.datadir + appliance_name + '/' + appliance_name + '_training_' + '.csv'
log('Training dataset: ' + training_path)

# Looking for the validation set
for filename in os.listdir(args.datadir + appliance_name):
    if "validation" in filename:
        val_filename = filename
        log(val_filename)

# path for validation data
validation_path = args.datadir + appliance_name + '/' + val_filename
log('Validation dataset: ' + validation_path)




# offset parameter from window length
#offset = int(0.5*(params_appliance[args.appliance_name]['windowlength']-1.0))

windowlength = 2053
#params_appliance[args.appliance_name]['windowlength']

# Defining object for training set loading and windowing provider (DataProvider.py)
tra_provider = ChunkS2S_Slider_fcn(filename=training_path,
                                        batchsize=args.batchsize,
                                        chunksize = CHUNK_SIZE,
                                        crop=args.crop_dataset,
                                        shuffle=True,
                                        length = windowlength,
                                        header=0,
                                        ram_threshold=args.ram)

# Defining object for validation set loading and windowing provider (DataProvider.py)
val_provider = ChunkS2S_Slider_fcn(filename=validation_path,
                                        batchsize=args.batchsize,
                                        chunksize=CHUNK_SIZE,
                                        crop=args.crop_dataset,
                                        shuffle=False,
                                        length = windowlength,
                                        header=0,
                                        ram_threshold=args.ram)

inputlength = windowlength + (windowlength//2)*2
# TensorFlow placeholders
x = tf.placeholder(tf.float32,
                   shape=[None, inputlength],
                   name='x')

y_ = tf.placeholder(tf.float32,
                    shape=[None, windowlength],
                    name='y_')

# -------------------------------- Keras Network - from model.py -----------------------------------------
inp = Input(tensor=x)
model = get_model(args.appliance_name,
                                     inp,
                                     inputlength,
                                     transfer_dense=args.transfer_model,
                                     transfer_cnn=args.transfer_cnn,
                                     cnn=args.cnn,
                                     pretrainedmodel_dir=args.pretrainedmodel_dir)
#cnn_check_weights
y = model.outputs

# -------------------------------------------------------------------------------------------------------

# cost function
cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(y, y_), 1))

# model's weights to be trained
train_params = tf.trainable_variables()
log("All network parameters: ")
log([v.name for v in train_params])
# if transfer learning is selected, just the dense layer will be trained
if not args.transfer_model and args.transfer_cnn:
    parameters = 10
else:
    parameters = 0
log("Trainable parameters:")
log([v.name for v in train_params[parameters:]])

# Training hyper parameters
train_op = tf.train.AdamOptimizer(learning_rate=0.001,
                                  beta1=0.9,
                                  beta2=0.999,
                                  epsilon=1e-08,
                                  use_locking=False).minimize(cost,
                                                              var_list=train_params[parameters:]
                                                              )

uninitialized_vars = []
for var in tf.all_variables():
    try:
        sess.run(var)
    except tf.errors.FailedPreconditionError:
        uninitialized_vars.append(var)


init_new_vars_op = tf.initialize_variables(uninitialized_vars)
sess.run(init_new_vars_op)


log('TensorFlow Session starting...')

# TensorBoard summary (graph)
tf.summary.scalar('cost', cost)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('./tensorboard_test')
writer.add_graph(sess.graph)
log('TensorBoard infos in ./tensorboard_test')

# Save path depending on the training behaviour
if not args.transfer_model and args.transfer_cnn:
    save_path = args.save_dir+'/cnn_s2p_' + appliance_name + '_transf_' + args.cnn + '_pointnet_model'
else:
    save_path = args.save_dir+'/cnn_s2p_' + appliance_name + '_pointnet_model'
    
if not os.path.exists(save_path):
        os.makedirs(save_path)

# Calling custom training function
train_loss, val_loss, step_train_loss, step_val_loss = nf.customfit(sess=sess,
                                                                    network=model,
                                                                    cost=cost,
                                                                    train_op=train_op,
                                                                    tra_provider=tra_provider,
                                                                    x=x,
                                                                    y_=y_,
                                                                    acc=None,
                                                                    n_epoch=args.n_epoch,
                                                                    print_freq=1,
                                                                    val_provider=val_provider,
                                                                    save_model=args.save_model,
                                                                    save_path=save_path,
                                                                    epoch_identifier=None,
                                                                    earlystopping=True,
                                                                    min_epoch=1,
                                                                    patience=25)

# Following are training info

log('train loss: ' + str(train_loss))
log('val loss: ' + str(val_loss))
infos = pd.DataFrame(data={'train_loss': step_train_loss,
                           #'val_loss': step_val_loss
                           })

plt.figure
epochs = range(1,len(train_loss)+1)
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('FCN_loss-{}.png'.format(appliance_name))
plt.show()
# infos = pd.DataFrame(data={'train_loss': train_loss,
#                            'val_loss': val_loss
#                            })
#
# infos.to_csv('./training_infos-{:}.csv'.format(appliance_name))
# # infos.to_csv('./training_infos-{:}-{:}-{:}.csv'.format(appliance_name, args.transfer, args.cnn))
# log('training infos in .csv file')


# This check that the CNN is the same of the beginning
# if not args.transfer_model and args.transfer_cnn:
#     log('Transfer learning check ...')
#     session = K.get_session()
#     for v in tf.trainable_variables():
#         if v.name == 'conv2d_1/kernel:0':
#             value = session.run(v)
#             vl = np.array(value).flatten()
#             c1 = np.array(cnn_check_weights).flatten()
#             if False in vl == c1:
#                 log('Transfer check --- ERROR ---')
#             else:
#                 log('Transfer check --- OK ---')


sess.close()





