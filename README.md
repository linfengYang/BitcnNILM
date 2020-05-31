# BitcnNILM
This code compare our method with Seq2point(Zhang), which proposed in Thirty-Second AAAI Conference on Articial Intelligence (AAAI-18), Feb. 2-7, 2018.  


**Requirements**

    1. Create your virtual environment Python > 3.5

    2. Install Tensorflow 2.0.0(default) or 1.4 < Tensorflow < 2.0.0 (You need to change our code a little bit) as following:

        * comment           -   import tensorflow.compat.v1 as tf 
                                tf.disable_v2_behavior()
                                session = tf.keras.backend.get_session()

        * And then uncomment -  import tensorflow as tf
                                session = K.get_session()
        
        * Remember a GPU support is highly recommended for training


    Install Keras > 2.1.5

    Clone this repository
    
# How to use the code and then reproduce our experiment. 
In this project, you can prepare the dataset, train the network and test it. 
    In REDD dataset, target appliances taken into account are microwave, fridge, dish washer and washing machine.
    In UK-DALE dataset, target appliances taken into account are kettle, microwave, fridge, dish washer and washing machine.
## **Create dataset**
You should select the following arguments for the argument parser:
`python create_dataset -h`

```
--data_dir DATA_DIR             The directory containing the CLEAN REDD OR UK-DALE data

--appliance_name APPLIANCE_NAME which appliance you want to train: kettle,
                                microwave,fridge,dishwasher,washingmachine

--aggregate_mean AGGREGATE_MEAN Mean value of aggregated reading (mains)

--aggregate_std AGGREGATE_STD   Std value of aggregated reading (mains)

`--save_path SAVE_PATH           The directory to store the training data
```


1. For experiment 1, create a REDD dataset (mains and appliance power measurments) for kettle:

`python create_trainset_redd.py --data_dir './' --appliance_name 'kettle' --aggregate_mean 522 --aggregate_std 814 --save_path'./'`

Download the REDD raw data from the original website (http://redd.csail.mit.edu/).
 

2. For experiment 2, create UK-DALE-2015 dataset

`python create_trainset_ukdale.py --data_dir './' --appliance_name 'kettle' --aggregate_mean 522 --aggregate_std 814 --save_path'./'`

Download the UK-DALE raw data from the original website (http://jack-kelly.com/data/). 



## **Train**
The seq2point_train.py(corresponding BitcnNILM_Model- Our model),seq2point_train_cnn.py(corresponding cnn_Model-Seq2point(Zhang) script are the entry points for the training phase. It loads the training dataset, including validation, and it starts the training.
It uses a script to load CSV dataset file into memory, prepares pairs of 599 samples aggregate data and 1 sample midpoint ground truth.
After randomly shuffle them, batches of BATCHSIZE size are input to the network for backpropagation purpose.
Once the training is cmplete, according to the eary stopping criterion, the trained KERAS model (and model's parameters) will be available into the folder you have selected.

Notice: the code of seq2point_train.py is same with seq2point_train_cnn.py except for importing different method. 

1. For experiment 1 based on REDD dataset, train our model and seq2point(Zhang) for (ex:dishwasher),respectively.
`python3 seq2point_train.py --appliance_name dishwasher --datadir ./dataset_management/redd/ --save_dir ./trained_model_BitcnNILM --transfer_model False > dishwasher.log`
`python3 seq2point_train_cnn.py --appliance_name dishwasher --datadir ./dataset_management/redd/ --save_dir ./trained_model_CNN --transfer_model False > dishwasher.log`

2. For experiment 2 based on UK-DALE dataset, train our model and seq2point(Zhang) for （ex:washingmachine),respectively.
`python3 seq2point_train.py --appliance_name washingmachine --datadir ../dataset_management/uk_no2/ --save_dir ./trained_model_BitcnNILM --transfer_model False > washingmachine.log`
`python3 seq2point_train_cnn.py --appliance_name washingmachine --datadir ../dataset_management/uk_no2/ --save_dir ./trained_model_CNN --transfer_model False > washingmachine.log`

## **Test**
The seq2point_test.py and seq2point.py script are the entry points for testing the network. In a similar way to the training windows are prepared, without shuffling, and sent to the network.
The prediction is stored and saved in .npy file together with aggregate and ground truth. If selected, the script will generate a plot (an example below).

1. For experiment 1 based on REDD dataset, test our model and seq2point(Zhang) in all time samples for (ex:washingmachine),respectively.
`python3 seq2point_test.py --appliance_name washingmachine --datadir ./dataset_management/redd/ --trained_model_dir ./trained_model_BitcnNILM --save_results_dir ./result --transfer False --plot_results True > washingmachine.log`
`python3 seq2point_test_cnn.py --appliance_name washingmachine --datadir ./dataset_management/redd/ --trained_model_dir ./trained_model_CNN --save_results_dir ./result --transfer False --plot_results True > washingmachine.log`

2. For experiment 2 based on UK-DALE dataset, test our model and seq2point(Zhang) in all time samples for (ex:washingmachine),respectively.
`python3 seq2point_test.py --appliance_name washingmachine --datadir ../dataset_management/uk_no2/ --trained_model_dir ./trained_model_BitcnNILM --save_results_dir ./result --transfer False --plot_results True > washingmachine.log`
`python3 seq2point_test_cnn.py --appliance_name washingmachine --datadir ../dataset_management/uk_no2/ --trained_model_dir ./trained_model_CNN --save_results_dir ./result --transfer False --plot_results True > washingmachine.log`


# Reference: 
Chaoyun Zhang, Mingjun Zhong, Zongzuo Wang, Nigel Goddard, and Charles Sutton. "Sequence-to-point learning with neural networks for nonintrusive load monitoring." Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18), Feb. 2-7, 2018. 

The code based on Seq2point(Zhang) for fair comparison. https://github.com/MingjunZhong/transferNILM/
