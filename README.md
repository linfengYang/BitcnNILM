# BitcnNILM
This code compare our method with Seq2point(Zhang), which proposed in Thirty-Second AAAI Conference on Articial Intelligence (AAAI-18), Feb. 2-7, 2018.  


1. Requirements

    Create your virtual environment Python > 3.5

    Install Tensorflow 2.0.0(default) or 1.4 < Tensorflow < 2.0.0 (You need to change our code a little bit) as following:

      comment           -       import tensorflow.compat.v1 as tf 
                                tf.disable_v2_behavior()
                                session = tf.keras.backend.get_session()

      and then uncomment -      import tensorflow as tf
                                session = K.get_session()


    Remember a GPU support is highly recommended for training


    Install Keras > 2.1.5

    Clone this repository
    
2. 

