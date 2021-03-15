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

from Logger import log
from DataProvider import DoubleSourceProvider3, DoubleSourceProvider_fcn, MultiApp_Slider
import NetFlowExt as nf
import nilm_metric as nm
from fcn_Model import get_model, weights_loader
import os
import numpy as np

###############################
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

##############################
from keras.layers import Input
import pandas as pd
import argparse
from Arguments import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    parser = argparse.ArgumentParser(description='Predict the appliance\
                                     give a trained neural network\
                                     for energy disaggregation -\
                                     network input = mains window;\
                                     network target = the states of\
                                     the target appliance.')
    parser.add_argument('--appliance_name',
                        type=remove_space,
                        default='kettle',
                        help='the name of target appliance')
    parser.add_argument('--datadir',
                        type=str,
                        default='/media/michele/Dati/myREFIT/',
                        help='this is the directory to the test data')
    parser.add_argument('--trained_model_dir',
                        type=str,
                        default='./models',
                        help='this is the directory to the trained models')
    parser.add_argument('--save_results_dir',
                        type=str,
                        default='./results',
                        help='this is the directory to save the predictions')
    parser.add_argument('--nosOfWindows',
                        type=int,
                        default=256, #256
                        help='The number of windows for prediction \
                            for each iteration.')
    parser.add_argument('--test_type',
                        type=str,
                        default='test',
                        help='Type of the test set to load: \
                            test -- test on the proper test set;\
                            train -- test on a aready prepared slice of the train set;\
                            val -- test on the validation set;\
                            uk -- test on UK-DALE;\
                            redd -- test on REDD.')
    parser.add_argument('--dense_layers',
                        type=int,
                        default=1,
                        help=':\
                                1 -- One dense layers (default Seq2point);\
                                2 -- Two dense layers;\
                                3 -- three dense layers the CNN.')
    parser.add_argument("--transfer", type=str2bool,
                        default=False,
                        help="Using a pre-trained CNN (True) or not (False).")
    parser.add_argument("--plot_results", type=str2bool,
                        default=False,
                        help="To plot the predicted appliance against ground truth or not.")
    parser.add_argument('--cnn',
                        type=str,
                        default='kettle',
                        help='The trained CNN by which appliance to load.')
    parser.add_argument('--crop_dataset',
                        type=int,
                        default=None,
                        help='for debugging porpose should be helpful to crop the test dataset size')
    return parser.parse_args()


args = get_arguments()
log('Arguments: ')
log(args)


def load_dataset(filename, offset,windowlength,header=0):
    data_frame = pd.read_csv(filename,skiprows= None, nrows=args.crop_dataset,header=header,na_filter=False)
    # data_frame = pd.read_csv(filename, nrows=None,header=header,na_filter=False)

    test_set_x = np.round(np.array(data_frame.iloc[:, 0], float), 5)

    num_windows = int(np.ceil((test_set_x.size - 2 * offset) / windowlength))
    # pad the end of the dataframe so that it has an exact multiple of windows
    pad_size = num_windows * windowlength + 2 * offset - test_set_x.size
    pad_zeros = np.zeros(pad_size)
    test_set_x = np.concatenate((test_set_x, pad_zeros), )

    test_set_y = np.round(np.array(data_frame.iloc[:, 1], float), 5)
    test_set_y = np.concatenate((test_set_y, pad_zeros), )
    #ground_truth = np.round(np.array(data_frame.iloc[offset:-offset, 1], float), 5)

    ground_truth = np.round(np.array(data_frame.iloc[:, 1], float), 5)
    ground_truth = np.concatenate((ground_truth, pad_zeros), )
    ground_truth = ground_truth[offset:-offset]
    #ground_truth = np.round(np.array(data_frame.iloc[0:-2*offset, 1]), 5)
    del data_frame
    return test_set_x, test_set_y, ground_truth


appliance_name = args.appliance_name
log('Appliance target is: ' + appliance_name)

# Looking for the selected test set
for filename in os.listdir(args.datadir + appliance_name):
        if args.test_type == 'train' and 'TRAIN' in filename.upper():
            test_filename = filename
            break
        elif args.test_type == 'uk' and 'UK' in filename.upper():
            test_filename = filename
            break
        elif args.test_type == 'redd' and 'REDD' in filename.upper():
            test_filename = filename
            break
        elif args.test_type == 'test' and 'TEST' in\
                filename.upper() and 'TRAIN' not in filename.upper() and 'UK' not in filename.upper():
            test_filename = filename
            break
        elif args.test_type == 'val' and 'VALIDATION' in filename.upper():
            test_filename = filename
            break


log('File for test: ' + test_filename)
loadname_test = args.datadir + appliance_name + '/' + test_filename
log('Loading from: ' + loadname_test)

# offset parameter from windowlenght
windowlength = 2053
#offset = int(0.5 * (params_appliance[args.appliance_name]['windowlength'] - 1.0))
offset = windowlength//2

test_set_x, test_set_y, ground_truth = load_dataset(loadname_test,offset,windowlength)
sess = tf.InteractiveSession()

# Dictonary containing the dataset input and target

test_kwag = {
    'inputs': test_set_x,
    'targets': test_set_y
}

# Defining object for training set loading and windowing provider
test_provider = DoubleSourceProvider_fcn(nofWindows=args.nosOfWindows,
                                      offset=offset,
                                      windowlength = windowlength)

# test_provider = MultiApp_Slider(batchsize = 1000, shuffle = False, offset=offset)
inputlength = windowlength + (windowlength//2)*2
# TensorFlow placeholders
x = tf.placeholder(tf.float32,
                   shape=[None, inputlength],
                   name='x')

# y_ = tf.placeholder(tf.float32,
#                     shape=[None, 1],
#                     name='y_')

# -------------------------------- Keras Network - from model.py -------------------------------------
inp = Input(tensor=x)

# model = get_model(args.appliance_name,
#                   inp,
#                   windowlength,
#                   )

model = get_model(args.appliance_name,
                  inp,
                  inputlength,
                  n_dense=args.dense_layers
                  )

y = model.outputs
# ----------------------------------------------------------------------------------------------------

sess.run(tf.global_variables_initializer())

# Load path depending on the model kind
if args.transfer:
    print('arg.transfer'.format(args.transfer))
    param_file = args.trained_model_dir+'/cnn_s2p_' + appliance_name + '_transf_' + args.cnn + '_pointnet_model'
else:
    print('arg.transfer'.format(args.transfer))
    param_file = args.trained_model_dir+'/cnn_s2p_' + args.appliance_name + '_pointnet_model'

# Loading weigths
log('Model file: {}'.format(param_file))
weights_loader(model, param_file)

# Calling custom test function
# test_prediction = nf.custompredict_add(sess=sess,
#                                    network=model,
#                                    output_provider = test_provider,
#                                    x = x,
#                                    fragment_size=args.nosOfWindows,
#                                    output_length=windowlength,
#                                    y_op= None,
#                                    out_kwag=test_kwag,seqlength = test_set_x.size)

test_prediction = nf.custompredict_fcn(sess=sess,
                                network=model,
                                output_provider=test_provider,
                                x=x,
                                fragment_size=args.nosOfWindows,
                                output_length=windowlength,
                                y_op=None,
                                out_kwag=test_kwag)


# ------------------------------------- Performance evaluation----------------------------------------------------------

# Parameters
max_power = params_appliance[args.appliance_name]['max_on_power']
threshold = params_appliance[args.appliance_name]['on_power_threshold']
aggregate_mean = 522
aggregate_std = 814


appliance_mean = params_appliance[args.appliance_name]['mean']
appliance_std = params_appliance[args.appliance_name]['std']

log('aggregate_mean: ' + str(aggregate_mean))
log('aggregate_std: ' + str(aggregate_std))
log('appliance_mean: ' + str(appliance_mean))
log('appliance_std: ' + str(appliance_std))
# prediction = test_prediction
#####
test_prediction = test_prediction.flatten()
#######
prediction = test_prediction * appliance_std + appliance_mean
prediction[prediction <= 0.0] = 0.0
ground_truth = ground_truth * appliance_std + appliance_mean
log('size: prediction={0}'.format(np.shape(prediction))) 
log('size: truth={0}'.format(np.shape(ground_truth)))

sess.close()


# ------------------------------------------ metric evaluation----------------------------------------------------------
sample_second = 6.0  # sample time is 6 seconds
on_off_metric = nm.recall_precision_accuracy_f1(prediction.flatten(), ground_truth.flatten(),threshold)
log("============ Recall: {}".format(on_off_metric[0]))
log("============ Precision: {}".format(on_off_metric[1]))
log("============ Accuracy: {}".format(on_off_metric[2]))
log("============ F1 Score: {}".format(on_off_metric[3]))

print("============ Recall: {}".format(on_off_metric[0]))
print("============ Precision: {}".format(on_off_metric[1]))
print("============ Accuracy: {}".format(on_off_metric[2]))
print("============ F1 Score: {}".format(on_off_metric[3]))

# log('F1:{0}'.format(nm.get_F1(ground_truth.flatten(), prediction.flatten(), threshold)))
# log('NDE:{0}'.format(nm.get_nde(ground_truth.flatten(), prediction.flatten())))
print("============")
print('\nMAE: {:}\n    -std: {:}\n    -min: {:}\n    -max: {:}\n    -q1: {:}\n    -median: {:}\n    -q2: {:}'
    .format(*nm.get_abs_error(ground_truth.flatten(), prediction.flatten())))
print('SAE: {:}'.format(nm.get_sae(ground_truth.flatten(), prediction.flatten(), sample_second)))
# log('Energy per Day: {:}'.format(nm.get_Epd(ground_truth.flatten(), prediction.flatten(), sample_second)))


# ----------------------------------------------- save results ---------------------------------------------------------
savemains = test_set_x.flatten() * aggregate_std + aggregate_mean
savegt = ground_truth
savepred = prediction.flatten()

if args.transfer:
    save_name = args.save_results_dir + '/' + appliance_name + '/' + test_filename + '_transf_' + args.cnn   # save path for mains
else:
    save_name = args.save_results_dir + '/' + appliance_name + '/' + test_filename  # save path for mains
if not os.path.exists(save_name):
        os.makedirs(save_name)
# Numpy saving
np.save(save_name + '_pred.npy', savepred)
np.save(save_name + '_gt.npy', savegt)
np.save(save_name + '_mains.npy', savemains)

"""
# saving in csv format
result_dict = {
    'aggregate': savepred,
    'ground truth': savegt,
    'prediction': savepred,
}

# CSV saving
result = pd.DataFrame(result_dict)
result.to_csv(save_name + '.csv', index=False)
"""

log('size: x={0}'.format(np.shape(savemains)))
log('size: y={0}'.format(np.shape(savepred)))
log('size: gt={0}'.format(np.shape(savegt)))

total = len(savemains)

# ----------------------------------------------- PLOT results ---------------------------------------------------------
if args.plot_results:
    ################
    # import matplotlib as mpl
    # mpl.use('Agg')
    ################
    import matplotlib.pyplot as plt

    if args.plot_results:
        fig1 = plt.figure()
        # plt.axis([0,total,0,12000])
        #plt.xticks([0,total,total])
        ax1 = fig1.add_subplot(111)

        ax1.plot(savemains, color='#7f7f7f', linewidth=1.8)
        # plt.show()
        # plt.savefig('{}-BiTCN-aggregate.png'.format(args.appliance_name))
        ax1.plot(ground_truth, color='#d62728', linewidth=1.6)
       
        # plt.savefig('{}-BiTCN0.png'.format(args.appliance_name))
        # plt.show()
        # plt.savefig('{}-BiTCN-truth.png'.format(args.appliance_name))
        ax1.plot(prediction, color='#1f77b4', linewidth=1.5)

        plt.xticks([])
        ax1.grid()
        # ax1.set_title('Test results on {:}'.format(test_filename), fontsize=16, fontweight='bold', y=1.08)
        ax1.set_ylabel(args.appliance_name.capitalize() + '\n' +'(Watt)')
        ax1.set_xlabel('Time(number of samples)')
        ax1.legend(['Aggregate', 'Ground Truth', 'FCN)'],loc='best')

        mng = plt.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        plt.savefig('{}-FCN.pdf'.format(args.appliance_name))
        plt.show(fig1)

        #subplot

        # plt.subplot(311)
        # plt.title(appliance_name)
        # plt.plot(savemains[offset:-offset])
        # # plt.axis([0,total,0,6000])
        # plt.yticks(np.linspace(0,6000,3,endpoint=True))
        # plt.xticks([0,200000,total])
        # plt.ylabel('Aggregate',fontsize = 10)
        #
        # plt.subplot(312)
        # plt.plot(ground_truth)
        # plt.axis([0,total,0,3000])
        # plt.yticks(np.linspace(0,3000,4,endpoint=True))
        # plt.xticks([0,200000,total])
        # plt.ylabel('Ground Truth',fontsize = 10)
        #
        # plt.subplot(313)
        # plt.plot(prediction)
        # plt.axis([0,total,0,3000])
        # plt.yticks(np.linspace(0,3000,4,endpoint=True))
        # plt.xticks([0,200000,total])
        # plt.ylabel('FCN',fontsize = 10)
        #
        # log('size: x={0}'.format(np.shape(savemains[offset:-offset])))
        # log('size: y={0}'.format(np.shape(savepred)))
        # log('size: gt={0}'.format(np.shape(savegt)))
        # plt.subplots_adjust(bottom=0.2, right=0.7, top=0.9, hspace=0.3)
        # plt.savefig('{}-FCN_subplot.pdf'.format(args.appliance_name))



